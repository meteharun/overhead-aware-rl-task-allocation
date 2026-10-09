"""The placement environment. One episode is one allocation, built task by task.

An episode walks an order of tasks and picks a core for each. With the fixed
order that order is the one the greedy packers use, so a comparison against bfd
or wfd isolates the core choice.

No feature is keyed by identity: there is no device id, no platform id and no
cpu type one hot anywhere. A core is described by its numbers, a crossing by
hop count, delay and cpu cost rather than by link type. So an unseen platform
is a different set of numbers in the same slots, which is what makes the
held-out platforms a real test. Core class survives as the one categorical
feature, because os, general and isolated are roles that exist everywhere.

The timing model is called once per episode, at the end. During the episode the
state is tracked with the same cheap load estimate the greedy packers use,
`(C + dispatch) / period x tail`, plus the link cpu a crossing adds at both
ends. Calling the real model per step costs about fifty times more.

The accounting below mirrors `allocators/common.py` rather than importing it,
so that file stays the sole definition of the baselines and of the dataset.
"""

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import tasksets  # noqa: E402
from timing import CRITICALITY, EXTERNAL_WEIGHT, Model, preference_penalty  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
PLATFORMS = ROOT / "data" / "platforms"

TASK_FEATURES = (
    "log_period", "deadline_ratio", "weight",
    "is_hard", "is_firm", "is_soft",
    "in_degree", "out_degree", "depth",
    "util_mean", "util_min", "util_max",
    "on_chain",
)

CORE_FEATURES = (
    "is_os", "is_general", "is_isolated",
    "tail", "wake", "dispatch", "preempt",
    "device_share", "device_links", "device_reach",
)

# Everything that depends on the partial allocation lives here rather than in the
# core encoder, so the core embedding can be computed once per instance.
PAIR_FEATURES = (
    "util",             # what this task would add to this core
    "load",             # what the core already carries
    "load_after",       # the sum, the number that decides whether it fits
    "overflow",         # how far past full it would go, 0 when it fits
    "preference",       # the objective's own class penalty, handed over directly
    "nbrs_here",        # placed neighbours already on this device
    "nbrs_away",        # placed neighbours that this placement would cut off
    "send_cost",        # link cpu this core would pay to send
    "elsewhere_cost",   # link cpu this placement dumps on other cores
    "delay",            # route delay to placed neighbours
    "hops",             # worst hop count to a placed neighbour
    "occupancy",        # tasks already on this core
    "tail",             # repeated here because it interacts with everything above
    "is_os",            # likewise: the one core class the packers refuse to use

    # Total load says nothing about whether this task meets its deadline. Under
    # deadline monotonic priorities only work with a SHORTER deadline delays
    # it, so a core at 0.8 load is harmless if all of it sits below this task
    # and fatal if all of it sits above.
    "higher_util",      # load on this core that outranks the task being placed
    "resp_ratio",       # first order response estimate, as a share of the deadline
    "would_miss",       # that estimate already past the deadline

    # The only feature about a constraint spanning the whole allocation.
    # Blowing a chain budget costs the weight of the heaviest task on the path.
    "chain_headroom",   # share of the chain budget still unspent if it goes here
)

N_TASK_FEATURES = len(TASK_FEATURES)
N_CORE_FEATURES = len(CORE_FEATURES)
N_PAIR_FEATURES = len(PAIR_FEATURES)


def _topological(task_ids, edges, key=None):
    """Placement order, dependencies first. `key` breaks ties among the tasks
    that are ready right now, exactly as in allocators/common.py."""
    indeg = {t: 0 for t in task_ids}
    succ = defaultdict(list)
    for s, d in edges:
        succ[s].append(d)
        indeg[d] += 1
    ready = deque(t for t in task_ids if indeg[t] == 0)
    order = []
    while ready:
        if key is None:
            n = ready.popleft()
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


class Instance:
    """One task set on one platform, with everything static precomputed.

    Building this is the expensive part of an episode, so `train.py` keeps a
    bounded cache of them.
    """

    def __init__(self, taskset, platform, order="topo"):
        self._setup(Model(taskset, platform), platform, order,
                    taskset["taskset_id"])

    @classmethod
    def from_model(cls, model, platform, order="topo"):
        """For the allocator wrapper, which is handed a Model it must reuse
        rather than build a second copy of."""
        inst = cls.__new__(cls)
        inst._setup(model, platform, order, "")
        return inst

    def _setup(self, model, platform, order, taskset_id):
        self.taskset_id = taskset_id
        self.platform_id = platform["platform_id"]
        self.platform = platform          # the baselines want the raw dict
        self.model = m = model

        self.cores = list(m.core)
        self.M = len(self.cores)
        self.core_index = {c: j for j, c in enumerate(self.cores)}

        # devices before tasks: the placement order depends on how many there are
        self.devices = [d["device_id"] for d in platform["devices"]]
        self.dev_index = {d: k for k, d in enumerate(self.devices)}
        self.core_device = np.array(
            [self.dev_index[m.core[c]["device_id"]] for c in self.cores])

        self.tasks = self._order(order)
        self.N = len(self.tasks)
        self.task_index = {t: n for n, t in enumerate(self.tasks)}

        self._routes(platform)
        self._matrices()
        self.task_feat = self._task_features()
        self.core_feat = self._core_features()
        self.adj_in, self.adj_out = self._adjacency()

    # ---- construction ------------------------------------------------------

    def _criticality_key(self):
        """Most expensive to miss first. The same key allocators/common.py uses,
        so `bfd_crit` and the policy are choosing from the same freedom."""
        m = self.model

        def key(tid):
            t = m.task[tid]
            return (-t["internal_weight"] * EXTERNAL_WEIGHT[t["period_us"]],
                    t["deadline_us"], tid)

        return key

    def _order(self, how):
        """Which task the policy is asked to place next.

        Topological, because the pair features describe where a task's
        already-placed neighbours are, which says nothing if they are not
        placed yet.

        On a single device every route is empty, so send cost, delay, hop count
        and "neighbours elsewhere" are identically zero and the topological
        order carries no information. There the ready set is ordered by what a
        miss would cost instead, as `bfd_crit` does. `topo_strict` keeps the
        topological order throughout, for the ablation.
        """
        m = self.model
        ids = list(m.task)
        if how == "deadline":
            return sorted(ids, key=lambda t: (m.task[t]["deadline_us"], t))
        if how == "util":
            def u(t):
                cs = [m.exec_us[m.task[t]["code"]][m.core[c]["cpu_type"]]
                      for c in self.cores]
                return sum(cs) / len(cs) / m.task[t]["period_us"]
            return sorted(ids, key=lambda t: (-u(t), t))
        key = (self._criticality_key()
               if how == "topo" and len(self.devices) == 1 else None)
        return _topological(ids, m.edges, key)

    def _routes(self, platform):
        """Per device pair: what a message costs and where it lands.

        `send` is the first hop's cpu cost, paid by the sending core. `landings`
        is every os core the message touches and what it costs there, including
        the extra send a relaying device pays. Stored as indices so the hot loop
        never touches a dictionary.
        """
        m = self.model
        D = len(self.devices)
        self.route_send = np.zeros((D, D))
        self.route_delay = np.zeros((D, D))
        self.route_hops = np.zeros((D, D))
        self.route_landings = [[[] for _ in range(D)] for _ in range(D)]
        self.route_total = np.zeros((D, D))

        for a, da in enumerate(self.devices):
            for b, db in enumerate(self.devices):
                if a == b:
                    continue
                hops = m.route(da, db)
                devs = m.route_devices(da, db)
                self.route_send[a, b] = hops[0]["cpu_us"]
                self.route_delay[a, b] = sum(h["delay_us"] for h in hops)
                self.route_hops[a, b] = len(hops)
                land, total = [], hops[0]["cpu_us"]
                for i, lk in enumerate(hops):
                    j = self.core_index[m.os_core[devs[i + 1]]]
                    land.append((j, lk["cpu_us"]))
                    total += lk["cpu_us"]
                    if i + 1 < len(hops):
                        land.append((j, hops[i + 1]["cpu_us"]))
                        total += hops[i + 1]["cpu_us"]
                self.route_landings[a][b] = land
                self.route_total[a, b] = total

    def _matrices(self):
        """[N, M] tables that never change: what each task costs on each core."""
        m = self.model
        N, M = self.N, self.M
        self.wcet = np.zeros((N, M))
        self.util = np.zeros((N, M))
        self.pref = np.zeros((N, M))
        self.period = np.array([m.task[t]["period_us"] for t in self.tasks], float)
        self.deadline = np.array([m.task[t]["deadline_us"] for t in self.tasks], float)
        # deadline as a share of period: what the first order response estimate
        # has to come in under
        self.deadline_frac = self.deadline / self.period

        for n, t in enumerate(self.tasks):
            code, p = m.task[t]["code"], m.task[t]["period_us"]
            for j, c in enumerate(self.cores):
                core = m.core[c]
                w = m.exec_us[code][core["cpu_type"]]
                self.wcet[n, j] = w
                self.util[n, j] = ((w + m.overhead[core["cpu_type"]]["dispatch"])
                                   / p * m.tail[core["core_class"]])
                self.pref[n, j] = preference_penalty(p, core["core_class"])

        self.core_tail = np.array([m.tail[m.core[c]["core_class"]] for c in self.cores])
        self.core_is_os = np.array(
            [1.0 if m.core[c]["core_class"] == "os" else 0.0 for c in self.cores])

        # neighbours in task-index space, so the step loop stays in integers
        self.succ = [[] for _ in range(N)]
        self.pred = [[] for _ in range(N)]
        for s, d in m.edges:
            self.succ[self.task_index[s]].append(self.task_index[d])
            self.pred[self.task_index[d]].append(self.task_index[s])
        self.degree = np.array([len(self.succ[n]) + len(self.pred[n])
                                for n in range(N)], float)

        # the chain the budget is measured over, and what it is allowed to cost
        self.on_chain = self._on_chain()
        self.chain_budget, self.chain_floor = self._chain_budget()

    def _adjacency(self):
        """Mean-aggregating adjacency, both directions.

        Dense N x N because N never exceeds a few dozen here, which is far
        cheaper than the bookkeeping a sparse representation would need.
        """
        N = self.N
        a_in = np.zeros((N, N), dtype=np.float32)
        a_out = np.zeros((N, N), dtype=np.float32)
        for n in range(N):
            for s in self.succ[n]:
                a_out[n, s] = 1.0
            for p in self.pred[n]:
                a_in[n, p] = 1.0
        a_in /= np.maximum(a_in.sum(1, keepdims=True), 1.0)
        a_out /= np.maximum(a_out.sum(1, keepdims=True), 1.0)
        return a_in, a_out

    def _chain_budget(self):
        """(budget, floor) of the first chain, or (0, 0) when there is none.

        Only the first: the generator writes one budget per task set, and the
        penalty charges each violated chain separately, so one number is the
        whole story here.
        """
        for b in self.model.budgets:
            budget, floor = b.get("budget_us"), b.get("floor_us")
            if budget:
                return float(budget), float(floor or 0.0)
        return 0.0, 0.0

    def _on_chain(self):
        """Tasks that sit between a budget's endpoints.

        Not the worst path, which depends on the allocation and so cannot be a
        static feature. Just everything the budget could possibly run through.
        """
        m = self.model
        flag = np.zeros(self.N)
        for b in m.budgets:
            src, dst = self.task_index.get(b["src"]), self.task_index.get(b["dst"])
            if src is None or dst is None:
                continue
            fwd, back = {src}, {dst}
            q = deque([src])
            while q:
                n = q.popleft()
                for s in self.succ[n]:
                    if s not in fwd:
                        fwd.add(s)
                        q.append(s)
            q = deque([dst])
            while q:
                n = q.popleft()
                for p in self.pred[n]:
                    if p not in back:
                        back.add(p)
                        q.append(p)
            for n in fwd & back:
                flag[n] = 1.0
        return flag

    def _task_features(self):
        m = self.model
        N = self.N
        f = np.zeros((N, N_TASK_FEATURES), dtype=np.float32)

        # relaxed in true topological order, which is not the placement order
        # unless --order topo was asked for
        depth = np.zeros(N)
        for tid in _topological(list(m.task), m.edges):
            n = self.task_index[tid]
            for s in self.succ[n]:
                depth[s] = max(depth[s], depth[n] + 1)
        depth = depth / max(depth.max(), 1.0)

        chain = self.on_chain
        raw_util = self.wcet / self.period[:, None]

        for n, t in enumerate(self.tasks):
            task = m.task[t]
            p = task["period_us"]
            crit = CRITICALITY[p]
            f[n] = [
                np.log10(p / 100.0) / 1.5,
                task["deadline_us"] / p,
                task["internal_weight"] / 2.0,
                float(crit == "HARD"), float(crit == "FIRM"), float(crit == "SOFT"),
                len(self.pred[n]) / 8.0, len(self.succ[n]) / 8.0, depth[n],
                raw_util[n].mean(), raw_util[n].min(), raw_util[n].max(),
                chain[n],
            ]
        return f

    def _core_features(self):
        m = self.model
        M = self.M
        f = np.zeros((M, N_CORE_FEATURES), dtype=np.float32)

        per_device = np.bincount(self.core_device, minlength=len(self.devices))
        # how connected this device is, and how far the rest of the platform is
        # from it. Topology facts, so a new wiring is a new number rather than
        # an unseen category.
        link_deg = (self.route_hops == 1).sum(1)
        reach = np.array([self.route_delay[k][self.route_delay[k] > 0].mean()
                          if (self.route_delay[k] > 0).any() else 0.0
                          for k in range(len(self.devices))])

        for j, c in enumerate(self.cores):
            core = m.core[c]
            ov = m.overhead[core["cpu_type"]]
            k = self.core_device[j]
            f[j] = [
                float(core["core_class"] == "os"),
                float(core["core_class"] == "general"),
                float(core["core_class"] == "isolated"),
                m.tail[core["core_class"]],
                ov["wake"] / 20.0, ov["dispatch"] / 20.0, ov["preempt"] / 20.0,
                per_device[k] / M,
                link_deg[k] / 3.0,
                np.log1p(reach[k]) / 10.0,
            ]
        return f


class Episode:
    """A partial allocation, and the pair features that describe extending it.

    Two things beyond building an allocation from nothing. A seed is placements
    that are already made and stay made, which is what repairing a failed core
    needs. Out of order placement lets a policy choose which task to place
    next: `todo` keeps the instance order for the fixed order case, `pending`
    is what is left, and the cursor skips anything already placed.
    """

    def __init__(self, inst, seed=None):
        self.inst = inst
        self.load = np.zeros(inst.M)
        self.n_on = np.zeros(inst.M)
        self.core_of = np.full(inst.N, -1, dtype=np.int64)
        # link receive work is an interrupt handler, so it outranks every task on
        # the core no matter what deadline that task has. Kept apart from `load`
        # because it counts as interference against everything.
        self.rx = np.zeros(inst.M)
        # [core, deadline, utilisation] per placed task. `entry` is indexed by
        # task so a later placement can correct a neighbour's cost; `entries`
        # holds the same objects compactly so the interference sum can iterate.
        self.entry = [None] * inst.N
        self.entries = []
        # link delay already committed between placed pairs that both sit on the
        # chain. The rest of the chain latency is periods and response times,
        # which the placement barely moves.
        self.chain_delay = 0.0

        self.todo = list(range(inst.N))
        self.pending = set(self.todo)
        self.cursor = 0

        for n, j in (seed or {}).items():
            self.place(int(n), int(j))

    def done(self):
        return not self.pending

    def next_task(self):
        """Fixed order: whatever comes next in the instance's own order."""
        return self.todo[self.cursor]

    def ready(self):
        """Learned order: every unplaced task whose predecessors are all placed.

        The same freedom `bfd_crit` has. Restricting to ready tasks keeps the
        neighbour features meaningful, since a task whose predecessors are
        unplaced has nothing to be pulled towards.
        """
        return [n for n in self.todo
                if n in self.pending
                and all(self.core_of[p] >= 0 for p in self.inst.pred[n])]

    def features(self, n=None):
        """[M, N_PAIR_FEATURES] for task `n`, or for the next one in order.

        Loops only over this task's already-placed neighbours, so the cost is
        degree times devices rather than anything quadratic.
        """
        inst = self.inst
        if n is None:
            n = self.next_task()
        M = inst.M
        dev = inst.core_device

        nbrs_here = np.zeros(M)
        send = np.zeros(M)
        elsewhere = np.zeros(M)
        delay = np.zeros(M)
        hops = np.zeros(M)
        chain_add = np.zeros(M)
        placed = 0
        on_chain = inst.on_chain[n] > 0

        p_self = inst.period[n]
        for s in inst.succ[n]:
            j = self.core_of[s]
            if j < 0:
                continue
            placed += 1
            k = dev[j]
            nbrs_here += (dev == k)
            # this task is the sender, so its own core pays the first hop
            send += inst.route_send[:, k][dev] / p_self
            elsewhere += (inst.route_total[:, k][dev] - inst.route_send[:, k][dev]) / p_self
            delay += inst.route_delay[:, k][dev]
            hops = np.maximum(hops, inst.route_hops[:, k][dev])
            if on_chain and inst.on_chain[s] > 0:
                chain_add += inst.route_delay[:, k][dev]

        for pr in inst.pred[n]:
            j = self.core_of[pr]
            if j < 0:
                continue
            placed += 1
            k = dev[j]
            nbrs_here += (dev == k)
            # the predecessor sends, so the first hop lands on its core, not this
            # one. Still a cost this placement causes, just somewhere else.
            elsewhere += inst.route_total[k, :][dev] / inst.period[pr]
            delay += inst.route_delay[k, :][dev]
            hops = np.maximum(hops, inst.route_hops[k, :][dev])
            if on_chain and inst.on_chain[pr] > 0:
                chain_add += inst.route_delay[k, :][dev]

        util = inst.util[n]
        after = self.load + util
        deg = max(inst.degree[n], 1.0)

        # Interference: receive handlers always, plus every placed task whose
        # deadline is shorter than this one's. One pass over what has been placed
        # so far, so an episode costs O(N^2/2) additions in total.
        d_n = inst.deadline[n]
        higher = self.rx.copy()
        for core, dl, u in self.entries:
            if dl < d_n:
                higher[core] += u

        # R/T ~ (C/T) / (1 - U_higher), the standard first order bound, against
        # this task's deadline as a share of its period, so above 1 means the
        # estimate already misses. Clipped, since past a point how badly it
        # misses carries no information.
        resp = util / np.maximum(1.0 - higher, 0.02)
        resp_ratio = np.minimum(resp / inst.deadline_frac[n], 5.0)

        # What is left of the chain budget once the periods along the path and
        # every hop already committed are paid for, plus what this placement
        # would add. Starts near 1 - 1/tightness and falls to 0 as hops eat it;
        # negative means this placement alone blows the budget. Zero for a task
        # the budget does not run through, and for a task set without one.
        if on_chain and inst.chain_budget > 0.0:
            left = (inst.chain_budget - inst.chain_floor
                    - self.chain_delay - chain_add) / inst.chain_budget
            chain_headroom = np.clip(left, -1.0, 1.0)
        else:
            chain_headroom = np.zeros(M)

        out = np.empty((M, N_PAIR_FEATURES), dtype=np.float32)
        out[:, 0] = util
        out[:, 1] = self.load
        out[:, 2] = after
        out[:, 3] = np.maximum(after - 1.0, 0.0)
        out[:, 4] = inst.pref[n]
        out[:, 5] = nbrs_here / deg
        out[:, 6] = (placed - nbrs_here) / deg
        out[:, 7] = send
        out[:, 8] = elsewhere
        out[:, 9] = np.log1p(delay) / 10.0
        out[:, 10] = hops
        out[:, 11] = self.n_on / max(inst.N / inst.M, 1.0)
        out[:, 12] = inst.core_tail
        out[:, 13] = inst.core_is_os
        # clipped for the same reason resp_ratio is: a core is just as unusable
        # at 3.0 as at 9.0, and an unbounded input would swamp the others
        out[:, 14] = np.minimum(higher, 3.0)
        out[:, 15] = resp_ratio
        out[:, 16] = (resp_ratio > 1.0)
        out[:, 17] = chain_headroom
        return out

    def place(self, n, j=None):
        """Put task `n` on core `j` and update the load estimate.

        Same accounting as the greedy packers: the task's own inflated
        utilisation, the sender's first hop, and the receive work on every os
        core the message passes through. Called as `place(j)` it places the
        next task in order, which is what the fixed order rollout wants.
        """
        inst = self.inst
        if j is None:
            n, j = self.next_task(), n

        self.core_of[n] = j
        self.load[j] += inst.util[n, j]
        self.n_on[j] += 1
        entry = [j, inst.deadline[n], inst.util[n, j]]
        self.entry[n] = entry
        self.entries.append(entry)
        here = inst.core_device[j]

        # The sender's own hop is part of its execution, so it raises this
        # task's cost on its own core rather than becoming interference. The
        # landings are handlers on another os core and outrank everything there,
        # so they go into `rx` as well as `load`.
        for s in inst.succ[n]:
            o = self.core_of[s]
            if o < 0:
                continue
            k = inst.core_device[o]
            if inst.on_chain[n] and inst.on_chain[s]:
                self.chain_delay += inst.route_delay[here, k]
            if k == here:
                continue
            send = inst.route_send[here, k] / inst.period[n]
            self.load[j] += send
            entry[2] += send                 # it is part of this task's own cost
            for c, cost in inst.route_landings[here][k]:
                self.load[c] += cost / inst.period[n]
                self.rx[c] += cost / inst.period[n]

        for pr in inst.pred[n]:
            o = self.core_of[pr]
            if o < 0:
                continue
            k = inst.core_device[o]
            if inst.on_chain[n] and inst.on_chain[pr]:
                self.chain_delay += inst.route_delay[k, here]
            if k == here:
                continue
            send = inst.route_send[k, here] / inst.period[pr]
            self.load[o] += send
            self.entry[pr][2] += send        # likewise, but for the predecessor
            for c, cost in inst.route_landings[k][here]:
                self.load[c] += cost / inst.period[pr]
                self.rx[c] += cost / inst.period[pr]

        self.pending.discard(n)
        while self.cursor < len(self.todo) and self.todo[self.cursor] not in self.pending:
            self.cursor += 1

    def allocation(self):
        inst = self.inst
        return {inst.tasks[n]: inst.cores[int(j)]
                for n, j in enumerate(self.core_of) if j >= 0}

    def score(self):
        """The real objective, from the timing model. Called once, at the end."""
        res = self.inst.model.evaluate(self.allocation())
        p = res["penalty"]
        return {"total": p["total"], "hard": p["misses"] + p["budgets"],
                "misses": p["misses"], "budgets": p["budgets"],
                "preference": p["preference"],
                "perfect": p["misses"] == 0 and p["budgets"] == 0}


_platform_cache = {}


def load_instance(taskset_id, order="topo"):
    ts = tasksets.load(taskset_id)
    pid = ts["generator"]["platform_id"]
    if pid not in _platform_cache:
        _platform_cache[pid] = json.loads((PLATFORMS / f"{pid}.json").read_text())
    return Instance(ts, _platform_cache[pid], order=order)
