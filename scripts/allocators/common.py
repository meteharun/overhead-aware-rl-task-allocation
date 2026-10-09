"""Shared body of the greedy allocators. Two core-selection rules live here.

`fit` is the bin packing rule. It chooses a device first, by where this task's
neighbours already are, then a core inside it: `bfd` takes the fullest core
with room, `wfd` takes the emptiest.

`eft` is earliest finish time, the rule HEFT uses. It puts every core on the
platform in one pool and takes the one where this task's response time comes
out lowest. Crossing a link raises the sender's effective execution, so a core
across a link is simply a slower core for this task.

Three rules both share:

1. Dependencies pull. A task prefers the device where its already-placed
   neighbours live, most neighbours first, then the main device, then the rest.
2. Crossings are charged to the sender's hop and to the receive load on every
   os core along the route. No core may go over full, wherever it sits.
3. The os core is reserved. Incoming messages land there as interrupt handlers
   above every task, so tasks go there only if nothing else has room.

Load is estimated as `(C + dispatch) / period x tail`, which does not need the
preemption count and therefore does not need an allocation to exist yet.
"""

from collections import Counter, defaultdict
from math import ceil

FULL = 1.0
MAX_ITER = 200


def core_response(model, core_id, items, receives, stop_on_miss=False):
    """Response time of every task on one core, deadline monotonic.

    The one copy of the response time equation on the allocator side. It must
    match `timing.py._response_times` exactly, or the admission test would
    simply move a miss from the allocator to the report.

    items    [(task_id, c_eff)] for the tasks on this core, any order
    receives [(c, period)] for the link receive handlers on this core

    Priority is decided here through the same `priority_rank` the scorer uses,
    so the order of `items` cannot change the priority assignment.

    Returns {task_id: (r, miss)}. With `stop_on_miss` the walk returns as soon
    as one task misses, so the dict is partial and only the verdict is valid.
    """
    from timing import priority_rank

    cpu = model.core[core_id]["cpu_type"]
    ov = model.overhead[cpu]
    tail = model.tail[model.core[core_id]["core_class"]]

    higher = list(receives)                      # interrupts outrank every task
    out = {}
    for tid, ci in sorted(items, key=lambda x: priority_rank(model.task, x[0])):
        di = model.task[tid]["deadline_us"]

        r = ov["wake"] + ci
        for _ in range(MAX_ITER):
            interference = sum(ceil(r / p) * (c + ov["dispatch"]) for c, p in higher)
            k = sum(max(0, ceil(r / p) - 1) for c, p in higher)
            nxt = (ov["wake"] + ci + interference + k * ov["preempt"]) * tail
            if abs(nxt - r) < 1e-9:
                break
            r = nxt
            if r > di:
                # past the deadline r can only grow, so the verdict is settled
                break

        out[tid] = (r, r > di)
        if out[tid][1] and stop_on_miss:
            return out
        higher.append((ci, model.task[tid]["period_us"]))

    return out


def schedulable(model, core_id, items, receives):
    """Would every task on this core still meet its deadline?

    An admission test: it refuses placements already known to fail rather than
    scoring candidates. Standard practice for partitioned fixed priority.
    """
    got = core_response(model, core_id, items, receives, stop_on_miss=True)
    return not any(miss for _, miss in got.values())


def topological(task_ids, edges, key=None):
    """Placement order. Dependencies first, always.

    Topological rather than deadline monotonic, because the rule that places a
    task near its neighbours only says anything once those neighbours are
    there. Ordering by deadline would scatter every chain.

    `key` breaks ties among the tasks that are ready, meaning every
    predecessor is already placed, so a packer can give the most critical
    available task first pick of the cores without losing the ordering.
    """
    indeg = {t: 0 for t in task_ids}
    succ = defaultdict(list)
    for s, d in edges:
        succ[s].append(d)
        indeg[d] += 1

    ready = [t for t in task_ids if indeg[t] == 0]
    order = []
    while ready:
        if key is None:
            n = ready.pop(0)
        else:
            n = min(ready, key=key)
            ready.remove(n)
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)

    seen = set(order)
    rest = [t for t in task_ids if t not in seen]
    return order + (sorted(rest, key=key) if key else rest)


def criticality_key(model):
    """Most expensive to miss first, then tightest deadline, then task id.

    The same weighting the objective uses, so a packer ordered by this is told
    what the score cares about.
    """
    from timing import EXTERNAL_WEIGHT

    def key(tid):
        t = model.task[tid]
        return (-t["internal_weight"] * EXTERNAL_WEIGHT[t["period_us"]],
                t["deadline_us"], tid)

    return key


def allocate(model, platform, spread=False, criticality=False, rule="fit",
             order_key=None, pinned=None):
    """One pass constructive placement.

    `rule` picks the core once the candidates are known.

      fit   the bin packing rule. Cheapest core class first, then the fullest
            core with room (`spread=False`) or the emptiest (`spread=True`).
            Candidates are drawn one device at a time, in neighbour order, so
            dependencies pull explicitly.

      eft   earliest finish time. Candidates are drawn from the whole platform
            at once and the core where this task's own response time comes out
            lowest wins, with preference and load breaking ties. Dependencies
            pull implicitly, since crossing a link raises the sender's
            effective execution and so its response time.

    `order_key` overrides the placement order. It is a key over task ids, used
    to choose among the tasks that are ready, exactly like `criticality`.

    `pinned` is {task_id: core_id} for tasks that are already running and must
    not move. They are committed as given and only the rest are placed, which
    is repair after a core fails.
    """
    devices = [d["device_id"] for d in platform["devices"]]
    load = {cid: 0.0 for cid in model.core}

    neighbours = defaultdict(set)
    for s, d in model.edges:
        neighbours[s].add(d)
        neighbours[d].add(s)

    def util(tid, cid):
        c, t = model.core[cid], model.task[tid]
        base = model.exec_us[t["code"]][c["cpu_type"]]
        return ((base + model.overhead[c["cpu_type"]]["dispatch"])
                / t["period_us"] * model.tail[c["core_class"]])

    def delta_for(tid, cid, alloc):
        """What this placement adds anywhere on the platform, not just here.

        Returns the load delta per core, the extra execution the sender now pays
        on each affected task, and the new receive handlers per os core.
        """
        delta, bump, new_rx = defaultdict(float), defaultdict(float), defaultdict(list)
        dev = model.core[cid]["device_id"]

        def charge(sender_core, sender_task, dev_a, dev_b, period):
            hops = model.route(dev_a, dev_b)
            if not hops:
                return
            delta[sender_core] += hops[0]["cpu_us"] / period
            bump[sender_task] += hops[0]["cpu_us"]
            route = model.route_devices(dev_a, dev_b)
            for i, lk in enumerate(hops):
                landing = model.os_core[route[i + 1]]
                delta[landing] += lk["cpu_us"] / period
                new_rx[landing].append((lk["cpu_us"], period))
                if i + 1 < len(hops):             # relaying, so it sends again
                    delta[landing] += hops[i + 1]["cpu_us"] / period
                    new_rx[landing].append((hops[i + 1]["cpu_us"], period))

        for dst in model.out_edges[tid]:
            if dst in alloc:
                charge(cid, tid, dev, model.core[alloc[dst]]["device_id"],
                       model.task[tid]["period_us"])
        for src in neighbours[tid] - set(model.out_edges[tid]):
            if src in alloc:
                charge(alloc[src], src, model.core[alloc[src]]["device_id"], dev,
                       model.task[src]["period_us"])

        delta[cid] += util(tid, cid)
        return delta, bump, new_rx

    def fits(tid, cid, alloc, delta, bump, new_rx):
        if any(load[c] + d > FULL for c, d in delta.items()):
            return False

        # every core this placement touches must still meet its deadlines
        touched = set(delta) | {alloc[t] for t in bump if t in alloc} | {cid}
        for c in touched:
            here = [(t, c_eff[t] + bump.get(t, 0.0))
                    for t in tasks_on[c] if t != tid]
            if c == cid:
                here.append((tid, c_eff_if(tid, cid, bump)))
            if not schedulable(model, c, here, receives_on[c] + new_rx.get(c, [])):
                return False
        return True

    def c_eff_if(tid, cid, bump):
        c = model.core[cid]
        return model.exec_us[model.task[tid]["code"]][c["cpu_type"]] + bump.get(tid, 0.0)

    def device_order(tid, alloc):
        """Where this task's neighbours already are, then the main device."""
        near = Counter(model.core[alloc[n]]["device_id"]
                       for n in neighbours[tid] if n in alloc)
        ranked = [d for d, _ in near.most_common()]
        return ranked + [d for d in devices if d not in ranked]

    def pick(options):
        # cheapest core class first, then the fit rule. The ordering comes from
        # the same penalty table the objective uses.
        from timing import preference_penalty
        return min(options, key=lambda o: (
            preference_penalty(model.task[o[2]]["period_us"],
                               model.core[o[0]]["core_class"]),
            load[o[0]] if spread else -load[o[0]]))

    def pick_eft(options):
        # lowest response time wins, preference and load only break ties
        from timing import preference_penalty
        return min(options, key=lambda o: (
            o[3],
            preference_penalty(model.task[o[2]]["period_us"],
                               model.core[o[0]]["core_class"]),
            load[o[0]]))

    def response_on(tid, cid, bump, new_rx):
        """What this task's response time would be if it went here."""
        here = [(t, c_eff[t] + bump.get(t, 0.0)) for t in tasks_on[cid] if t != tid]
        here.append((tid, c_eff_if(tid, cid, bump)))
        got = core_response(model, cid, here,
                            receives_on[cid] + new_rx.get(cid, []))
        return got[tid][0]

    def candidates(tid, devs, allow_os):
        out = []
        for dev in devs:
            for cid in model.cores_of_device[dev]:
                if not allow_os and model.core[cid]["core_class"] == "os":
                    continue
                d, b, nrx = delta_for(tid, cid, alloc)
                if not fits(tid, cid, alloc, d, b, nrx):
                    continue
                r = response_on(tid, cid, b, nrx) if rule == "eft" else None
                out.append((cid, (d, b, nrx), tid, r))
        return out

    alloc = {}
    c_eff = {}
    tasks_on = defaultdict(list)
    receives_on = defaultdict(list)

    def commit(tid, cid, payload=None):
        """Put a task on a core and charge everything that costs."""
        delta, bump, new_rx = payload or delta_for(tid, cid, alloc)
        alloc[tid] = cid
        tasks_on[cid].append(tid)
        c_eff[tid] = c_eff_if(tid, cid, bump)
        for t, extra in bump.items():
            if t != tid and t in c_eff:
                c_eff[t] += extra
        for c, rs in new_rx.items():
            receives_on[c] += rs
        for c, d in delta.items():
            load[c] += d

    # Repair. Pinned tasks are committed without being offered a choice. Their
    # order does not matter: a crossing is charged when its second end lands,
    # so every edge is paid for exactly once however they arrive.
    todo = list(model.task)
    edges = model.edges
    if pinned:
        for tid, cid in pinned.items():
            commit(tid, cid)
        todo = [t for t in todo if t not in pinned]
        left = set(todo)
        # an edge to a pinned task is already resolved, and would otherwise be
        # an indegree the ordering could never discharge
        edges = [(a, b) for a, b in edges if a in left and b in left]

    key = order_key or (criticality_key(model) if criticality else None)
    for tid in topological(todo, edges, key):
        chosen = None

        # os cores are for interrupts, so consider them only once nothing else
        # on the platform has room
        for allow_os in (False, True):
            if rule == "eft":
                # every core competes at once, so the response time decides
                # rather than the device ordering
                options = candidates(tid, devices, allow_os)
                if options:
                    chosen = pick_eft(options)
            else:
                for dev in device_order(tid, alloc):
                    options = candidates(tid, [dev], allow_os)
                    if options:
                        chosen = pick(options)
                        break
            if chosen:
                break

        if chosen is None:
            # nothing can take it without a miss. No placement is ever
            # forbidden, so use the emptiest non-os core and let the penalty
            # record the miss. The same fallback for every rule, so none of
            # the comparison sits in the fallback.
            pool = [c for c in model.core if model.core[c]["core_class"] != "os"]
            cid = min(pool, key=lambda c: load[c] + util(tid, c))
            chosen = (cid, delta_for(tid, cid, alloc), tid, None)

        cid, payload, _, _ = chosen
        commit(tid, cid, payload)

    return alloc
