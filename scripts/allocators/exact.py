"""Exact allocator. Branch and bound over every assignment, scored by timing.py.

The optimality reference. Where this finishes, its answer is proved optimal
rather than merely the best anything found.

Branch and bound rather than a solver, because the constraint that decides
whether an allocation works is the response time recurrence in `timing.py`, a
fixed point over ceilings. Encoding it in an MILP or CP model means
approximating it, which would be exactly optimal for a model that is not the
thesis model. Calling the real model keeps the answer attached to the same
analysis every other allocator is judged by.

The pruning is valid because extending a partial allocation can only make
things worse. Another task on a core adds interference and takes nothing away;
a new crossing adds link cpu cost to the sender and a receive handler to an os
core and removes neither. So response times are non decreasing, a miss stays a
miss, and chain latency only grows. The penalty of the tasks placed so far is
therefore a lower bound on every completion, and a subtree whose bound has
reached the incumbent cannot contain anything better. Without that the search
is 28^60.

Symmetry: among empty cores of the same class and cpu type on one device only
the first is tried, and likewise among structurally identical empty devices.
On `platform_19`, four identical devices, this is the difference between
running and not running.

It is exact, so it is exponential. Small instances finish in seconds; a sixty
task, twenty eight core instance hits the time limit and returns the best
allocation found plus a lower bound. `exact_optimal` says which happened, and
a mean over a pool should not mix the two silently.

The objective defaults to `hard`, misses plus chain budget overruns, which is
what the results table compares on and which lets the search stop the moment
it reaches zero. Pass `objective="total"` for the preference weighted version
simulated annealing minimises.
"""

import time
from collections import defaultdict

from timing import EXTERNAL_WEIGHT, preference_penalty, worst_path

from .common import core_response, criticality_key

TIME_LIMIT = 10.0        # seconds per instance
NODE_LIMIT = None        # placements tried, None for no limit
OBJECTIVE = "hard"       # "hard" is misses plus budgets, "total" adds preference

# The incumbent to start from, cheapest allocator first. The walk stops as soon
# as one reaches zero, so the expensive entries are only paid on hard
# instances. Where the search proves optimality the starting point changes only
# how fast it gets there, not the answer.
WARM = ("bfd", "wfd", "bfd_crit", "wfd_crit", "heft", "heft_crit", "rl")

# the best allocation already on disk, read as the opening incumbent instead of
# rerunning the search that produced it
WARM_FILE = "sa_rl"

_warm_cache = {}


def _from_file(name, taskset_id):
    """The saved allocation for this instance, if there is one."""
    if name not in _warm_cache:
        from pathlib import Path
        import json
        f = (Path(__file__).resolve().parent.parent.parent
             / "data" / "allocations" / f"{name}.jsonl")
        got = {}
        if f.exists():
            with f.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    got[r["taskset_id"]] = r["allocation"]
        _warm_cache[name] = got
    return _warm_cache[name].get(taskset_id)


class Search:
    def __init__(self, model, platform, objective=OBJECTIVE,
                 time_limit=TIME_LIMIT, node_limit=NODE_LIMIT, warm=WARM,
                 warm_file=WARM_FILE, warm_alloc=None):
        self.m = model
        self.platform = platform
        self.objective = objective
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.warm = warm
        self.warm_file = warm_file
        # an allocation handed in directly, for a caller that already knows the
        # best one available, including from something not in the registry
        self.warm_alloc = warm_alloc

        m = model
        self.weight = {
            t: m.task[t]["internal_weight"] * EXTERNAL_WEIGHT[m.task[t]["period_us"]]
            for t in m.task}

        self.in_edges = defaultdict(list)
        for s, d in m.edges:
            self.in_edges[d].append(s)

        # heaviest to miss first, so the bound learns something immediately.
        # Leaving the expensive tasks last keeps the bound at zero through most
        # of the tree and prunes nothing.
        self.order = sorted(m.task, key=criticality_key(m))

        self.cores = list(m.core)
        self.dev_sig = self._device_signatures()
        self.cones = [(b, self._cone(b["src"], b["dst"])) for b in m.budgets]

        # a task that cannot meet its deadline alone on an idle core misses
        # wherever it is put, so the bound charges its weight from the root
        self.doomed = {t for t in m.task if self._always_misses(t)}

        self.alloc = {}
        self.on_core = defaultdict(list)
        self.c_eff = {}
        self.rx = defaultdict(list)
        self.r = {}
        self.missed = set()
        self.pref = 0.0

    # ---- structure, computed once -----------------------------------------

    def _device_signatures(self):
        """What makes two devices interchangeable.

        Same cores, and the same links on the route to every other device. In a
        star two leaves of the same board wired the same way match, and the hub
        never matches a leaf because its routes are shorter.
        """
        m = self.m
        devs = sorted(m.cores_of_device)
        sig = {}
        for d in devs:
            cores = tuple(sorted((m.core[c]["cpu_type"], m.core[c]["core_class"])
                                 for c in m.cores_of_device[d]))
            routes = tuple(sorted(
                tuple(lk["link_type"] for lk in m.route(d, o))
                for o in devs if o != d))
            sig[d] = (cores, routes)
        return sig

    def _always_misses(self, tid):
        """Does this task miss on every core even with the core to itself?

        The cheapest it can possibly be is its own execution, one wake and the
        class tail. Over the deadline on every core means no allocation saves it.
        """
        m = self.m
        code = m.task[tid]["code"]
        for c in m.core.values():
            r = (m.overhead[c["cpu_type"]]["wake"] + m.exec_us[code][c["cpu_type"]]) \
                * m.tail[c["core_class"]]
            if r <= m.task[tid]["deadline_us"]:
                return False
        return True

    def _cone(self, src, dst):
        """Tasks that lie on some path from src to dst.

        The chain term only enters the bound once all of them are placed, since
        the longest path is not known before that.
        """
        m = self.m
        rev = defaultdict(list)
        for s, d in m.edges:
            rev[d].append(s)

        def reach(start, adj):
            seen, stack = set(), [start]
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                stack.extend(adj.get(n, ()))
            return seen

        return reach(src, m.adj) & reach(dst, rev)

    # ---- incremental state -------------------------------------------------

    def _charge(self, sender_core, sender_task, dev_a, dev_b, period, bump, new_rx):
        """One crossing, priced exactly the way timing.py prices it."""
        m = self.m
        hops = m.route(dev_a, dev_b)
        if not hops:
            return
        bump[sender_task] = bump.get(sender_task, 0.0) + hops[0]["cpu_us"]
        route = m.route_devices(dev_a, dev_b)
        for i, lk in enumerate(hops):
            landing = m.os_core[route[i + 1]]
            new_rx[landing].append((lk["cpu_us"], period))
            if i + 1 < len(hops):              # relaying, so it sends again
                new_rx[landing].append((hops[i + 1]["cpu_us"], period))

    def place(self, tid, cid):
        m = self.m
        dev = m.core[cid]["device_id"]
        bump, new_rx = {}, defaultdict(list)

        for dst in m.out_edges[tid]:
            if dst in self.alloc:
                self._charge(cid, tid, dev, m.core[self.alloc[dst]]["device_id"],
                             m.task[tid]["period_us"], bump, new_rx)
        for src in self.in_edges[tid]:
            if src in self.alloc:
                self._charge(self.alloc[src], src,
                             m.core[self.alloc[src]]["device_id"], dev,
                             m.task[src]["period_us"], bump, new_rx)

        own = bump.pop(tid, 0.0)
        self.alloc[tid] = cid
        self.on_core[cid].append(tid)
        self.c_eff[tid] = m.exec_us[m.task[tid]["code"]][m.core[cid]["cpu_type"]] + own
        # store the old value, not the delta: adding a float and subtracting it
        # again does not always land back where it started, and the undo has to
        # restore the state exactly
        old_c_eff = {t: self.c_eff[t] for t in bump}
        for t, extra in bump.items():
            self.c_eff[t] += extra
        for c, rs in new_rx.items():
            self.rx[c].extend(rs)

        add_pref = preference_penalty(m.task[tid]["period_us"],
                                      m.core[cid]["core_class"])
        self.pref += add_pref

        touched = {cid} | {self.alloc[t] for t in bump} | set(new_rx)
        old_r, new_missed = {}, set()
        for c in touched:
            items = [(t, self.c_eff[t]) for t in self.on_core[c]]
            for t, (r, miss) in core_response(m, c, items, self.rx[c]).items():
                if t not in old_r:
                    old_r[t] = self.r.get(t)
                self.r[t] = r
                if miss and t not in self.missed:
                    self.missed.add(t)
                    new_missed.add(t)

        return (tid, cid, old_c_eff, {c: len(v) for c, v in new_rx.items()},
                old_r, new_missed, add_pref)

    def unplace(self, u):
        tid, cid, old_c_eff, rx_added, old_r, new_missed, add_pref = u
        for t, was in old_c_eff.items():
            self.c_eff[t] = was
        del self.c_eff[tid]
        for c, n in rx_added.items():
            del self.rx[c][len(self.rx[c]) - n:]
        self.on_core[cid].pop()          # strictly last in, depth first
        del self.alloc[tid]
        self.missed -= new_missed
        for t, r in old_r.items():
            if r is None:
                self.r.pop(t, None)
            else:
                self.r[t] = r
        self.pref -= add_pref

    # ---- the bound ---------------------------------------------------------

    def chain_penalty(self):
        m = self.m
        total = 0.0
        for b, cone in self.cones:
            if b.get("budget_us") is None or not cone.issubset(self.alloc):
                continue

            def node_cost(n):
                return m.task[n]["period_us"] + self.r[n]

            def edge_cost(a, c):
                return sum(lk["delay_us"] for lk in m.route(
                    m.core[self.alloc[a]]["device_id"],
                    m.core[self.alloc[c]]["device_id"]))

            found = worst_path(m.adj, b["src"], b["dst"], node_cost, edge_cost)
            if found is None:
                continue
            if self.r[b["src"]] + found[0] > b["budget_us"]:
                total += max(self.weight[n] for n in found[1])
        return total

    def bound(self):
        v = sum(self.weight[t] for t in self.missed) + self.chain_penalty()
        # a doomed task that is already placed is in `missed` and must not be
        # counted twice
        v += sum(self.weight[t] for t in self.doomed if t not in self.alloc)
        return v + self.pref if self.objective == "total" else v

    def cost_of(self, res):
        p = res["penalty"]
        return p["total"] if self.objective == "total" else p["misses"] + p["budgets"]

    # ---- branching ---------------------------------------------------------

    def core_order(self, tid):
        """Which cores to try, and in what order.

        Symmetric cores are dropped, which is safe because any solution using
        the skipped core maps to an equal cost solution using the kept one.

        The order affects only speed. The incumbent's own choice goes first, so
        the search re-derives a known good allocation early.
        """
        m = self.m
        out, seen_dev, seen_core = [], set(), set()
        for cid in self.cores:
            d = m.core[cid]["device_id"]
            if not any(self.on_core[c] for c in m.cores_of_device[d]):
                if self.dev_sig[d] in seen_dev:
                    continue
                seen_dev.add(self.dev_sig[d])
            if not self.on_core[cid]:
                sig = (d, m.core[cid]["cpu_type"], m.core[cid]["core_class"])
                if sig in seen_core:
                    continue
                seen_core.add(sig)
            out.append(cid)

        inc = self.best_alloc.get(tid) if self.best_alloc else None
        period = m.task[tid]["period_us"]

        def load(cid):
            return (sum(self.c_eff[t] / m.task[t]["period_us"]
                        for t in self.on_core[cid])
                    + sum(c / p for c, p in self.rx[cid]))

        out.sort(key=lambda c: (0 if c == inc else 1,
                                preference_penalty(period, m.core[c]["core_class"]),
                                load(c)))
        return out

    def _dfs(self, k):
        if self.best <= 0.0:
            return                                  # nothing can beat no penalty
        if k == len(self.order):
            # the incremental bound drives the pruning, the real model decides
            # what gets recorded, so a bound that drifted from timing.py would
            # cost optimality rather than correctness of the reported number
            cost = self.cost_of(self.m.evaluate(self.alloc))
            if cost < self.best:
                self.best, self.best_alloc = cost, dict(self.alloc)
            return
        if self.stop():
            self.open_lb = min(self.open_lb, self.bound())
            return

        tid = self.order[k]
        for cid in self.core_order(tid):
            self.nodes += 1
            u = self.place(tid, cid)
            if self.bound() < self.best:
                self._dfs(k + 1)
            self.unplace(u)
            if self.cut or self.best <= 0.0:
                if self.cut:
                    self.open_lb = min(self.open_lb, self.bound())
                return

    def stop(self):
        if self.cut:
            return True
        if self.node_limit is not None and self.nodes >= self.node_limit:
            self.cut = True
        elif time.perf_counter() >= self.until:
            self.cut = True
        return self.cut

    # ---- entry point -------------------------------------------------------

    def run(self):
        from . import get

        t0 = time.perf_counter()
        self.best, self.best_alloc = float("inf"), None

        # the best allocation already known costs one evaluation instead of a
        # search. Handed in beats read from a file.
        saved = self.warm_alloc or (
            _from_file(self.warm_file, getattr(self.m, "taskset_id", None))
            if self.warm_file else None)
        if saved and set(saved) == set(self.m.task) and \
                all(c in self.m.core for c in saved.values()):
            self.best = self.cost_of(self.m.evaluate(saved))
            self.best_alloc = dict(saved)

        for name in self.warm:
            if self.best <= 0.0:
                break                               # cannot be beaten, stop paying
            try:
                a = get(name)(self.m, self.platform)
            except SystemExit:
                continue                            # allocator not registered
            c = self.cost_of(self.m.evaluate(a))
            if c < self.best:
                self.best, self.best_alloc = c, a
        warm_cost = self.best

        self.nodes, self.cut = 0, False
        self.open_lb = float("inf")
        self.until = t0 + self.time_limit
        self._dfs(0)

        lb = min(self.best, self.open_lb)
        # Three outcomes, and they are three different facts.
        #
        #   schedulable  a zero penalty allocation was found. Zero is the floor
        #                of the hard objective, so this is optimal and the
        #                instance is provably schedulable.
        #   infeasible   the tree was explored to the end and the best is still
        #                above zero, which proves no allocation of this task set
        #                on this platform meets every deadline and budget.
        #   timeout      ran out of time. The allocation is the best found and
        #                the instance's status is unknown.
        proof = "timeout"
        if not self.cut:
            proof = "schedulable" if self.best <= 0.0 else "infeasible"
        self.stats = {
            "exact_objective": self.objective,
            "exact_optimal": not self.cut,
            "exact_status": proof,
            "exact_cost": round(self.best, 4),
            "exact_lower_bound": round(lb, 4),
            "exact_warm_cost": round(warm_cost, 4),
            "exact_improved": round(warm_cost - self.best, 4),
            "exact_nodes": self.nodes,
            "exact_ms": round(1000 * (time.perf_counter() - t0), 2),
        }
        return self.best_alloc


def make(objective=OBJECTIVE, time_limit=TIME_LIMIT, node_limit=NODE_LIMIT,
         warm=WARM, warm_file=WARM_FILE, name="exact"):
    """An exact allocator with these settings, registered like any other."""
    def allocate(model, platform):
        s = Search(model, platform, objective, time_limit, node_limit, warm,
                   warm_file, getattr(model, "warm_alloc", None))
        alloc = s.run()
        allocate.last = s.stats
        return alloc

    allocate.last = None
    allocate.__name__ = name
    allocate.__doc__ = (f"branch and bound on penalty_{objective}, "
                        f"{time_limit}s per instance")
    return allocate
