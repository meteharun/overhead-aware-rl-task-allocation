"""Timing model. Scores one allocation of a task set onto a platform.

Nothing here is stored on disk. Criticality, priority, whether an edge crosses a
link, and the paths of a chain are all derived from the allocation every time.

See README.md, "The timing model", for why it is shaped this way.
"""

import json
from collections import defaultdict
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOGUES = ROOT / "data" / "catalogues"

# derived from period, never stored
EXTERNAL_WEIGHT = {100: 100, 500: 10, 1000: 1, 2000: 1}
CRITICALITY = {100: "HARD", 500: "FIRM", 1000: "SOFT", 2000: "SOFT"}
# Charged once per task for the core class it ended up on. A HARD task pays to
# leave an isolated core, more the further it goes; anything else pays for
# taking an isolated seat, less for a general one.
PREFERENCE_PENALTY = {
    "HARD": {"isolated": 0.00, "general": 0.25, "os": 0.50},
    "FIRM": {"isolated": 0.25, "general": 0.10, "os": 0.00},
    "SOFT": {"isolated": 0.25, "general": 0.10, "os": 0.00},
}


def preference_penalty(period_us, core_class):
    return PREFERENCE_PENALTY[CRITICALITY[period_us]][core_class]


def priority_rank(task, tid):
    """Deadline monotonic, ties broken by task id. Lower sorts higher priority.

    Deadlines are integers in a narrow band, so a core often holds two tasks
    with the same one and the tie-break decides their order. Everything that
    assigns priority sorts through here: the scorer below, the admission test
    in `allocators/common.py`, and the simulator in `viz_alloc.py`.
    """
    return (task[tid]["deadline_us"], tid)


MAX_ITER = 1000


def load_catalogues():
    ex = json.loads((CATALOGUES / "exec_table.json").read_text())["exec_us"]
    hw = json.loads((CATALOGUES / "hardware_catalogue.json").read_text())
    return ex, hw


def worst_path(adj, src, dst, node_cost, edge_cost):
    """Longest path from src to dst in a DAG, by node and edge cost.

    Returns (total, path) with src's own cost excluded, or None if dst is
    unreachable. Memoised, so it does not enumerate every simple path.
    """
    memo = {}

    def go(n):
        if n in memo:
            return memo[n]
        if n == dst:
            memo[n] = (0.0, [dst])
            return memo[n]
        best = None
        for m in adj.get(n, ()):
            sub = go(m)
            if sub is None:
                continue
            v = edge_cost(n, m) + node_cost(m) + sub[0]
            if best is None or v > best[0]:
                best = (v, [n] + sub[1])
        memo[n] = best
        return best

    return go(src)


class Model:
    def __init__(self, taskset, platform, exec_us=None, hw=None):
        if exec_us is None or hw is None:
            exec_us, hw = load_catalogues()
        self.exec_us = exec_us
        self.tail = hw["core_class_tail"]
        self.overhead = hw["cpu_overheads_us"]
        self.link_tbl = {lt["link_type"]: lt for lt in hw["link_types"]}

        self.task = {t["task_id"]: t for t in taskset["tasks"]}
        self.edges = [(e["src"], e["dst"]) for e in taskset["edges"]]
        self.budgets = taskset["budgets"]

        self.adj = defaultdict(list)
        self.out_edges = defaultdict(list)
        for s, d in self.edges:
            self.adj[s].append(d)
            self.out_edges[s].append(d)

        self.core = {c["core_id"]: c for c in platform["cores"]}
        self.cores_of_device = defaultdict(list)
        for c in platform["cores"]:
            self.cores_of_device[c["device_id"]].append(c["core_id"])

        # every device has exactly one os core, and it is where receive work lands
        self.os_core = {}
        for dev, ids in self.cores_of_device.items():
            os_ids = [i for i in ids if self.core[i]["core_class"] == "os"]
            if len(os_ids) != 1:
                raise ValueError(f"{dev} has {len(os_ids)} os cores, expected exactly 1")
            self.os_core[dev] = os_ids[0]

        self.link = {}
        neighbours = defaultdict(list)
        for lk in platform["links"]:
            self.link[frozenset((lk["dev_a"], lk["dev_b"]))] = lk["link_type"]
            neighbours[lk["dev_a"]].append(lk["dev_b"])
            neighbours[lk["dev_b"]].append(lk["dev_a"])

        self._routes = {}
        for a in self.cores_of_device:
            for b in self.cores_of_device:
                self._routes[(a, b)] = self._shortest(a, b, neighbours)

    def _shortest(self, src, dst, neighbours):
        """Device hops from src to dst. Empty when they are the same device.

        Devices with no direct link are reached through a relay, and every hop
        is charged separately.
        """
        if src == dst:
            return []
        seen, queue = {src}, [[src]]
        while queue:
            path = queue.pop(0)
            for nxt in neighbours[path[-1]]:
                if nxt in seen:
                    continue
                if nxt == dst:
                    return path + [nxt]
                seen.add(nxt)
                queue.append(path + [nxt])
        raise ValueError(f"no route from {src} to {dst}")

    # ---- lookups -----------------------------------------------------------

    def route(self, dev_a, dev_b):
        """The links a message traverses, in order. Empty on the same device."""
        hops = self._routes[(dev_a, dev_b)]
        return [self.link_tbl[self.link[frozenset((x, y))]]
                for x, y in zip(hops, hops[1:])]

    def route_devices(self, dev_a, dev_b):
        return self._routes[(dev_a, dev_b)]

    def device_of(self, task_id, alloc):
        return self.core[alloc[task_id]]["device_id"]

    # ---- the model ---------------------------------------------------------

    def _effective_execution(self, alloc):
        """C_eff per task, plus the receive work each os core picks up.

        The sending task pays the cpu cost of the first hop only. Every device
        the message lands on pays a receive on its os core, and a device that is
        only relaying also pays the send cost of the next hop.
        """
        c_eff, receives = {}, defaultdict(list)

        for tid, t in self.task.items():
            cpu = self.core[alloc[tid]]["cpu_type"]
            total = self.exec_us[t["code"]][cpu]
            src_dev = self.device_of(tid, alloc)

            for dst in self.out_edges[tid]:
                dst_dev = self.device_of(dst, alloc)
                hops = self.route(src_dev, dst_dev)
                if not hops:
                    continue

                total += hops[0]["cpu_us"]          # sender pays the first hop only
                devs = self.route_devices(src_dev, dst_dev)

                for i, lk in enumerate(hops):
                    landing = self.os_core[devs[i + 1]]
                    receives[landing].append(
                        {"c": lk["cpu_us"], "period": t["period_us"],
                         "edge": (tid, dst), "kind": "recv",
                         "link_type": lk["link_type"]})
                    if i + 1 < len(hops):           # relaying, so it sends again
                        receives[landing].append(
                            {"c": hops[i + 1]["cpu_us"], "period": t["period_us"],
                             "edge": (tid, dst), "kind": "fwd",
                             "link_type": hops[i + 1]["link_type"]})

            c_eff[tid] = total

        return c_eff, receives

    def _response_times(self, alloc, c_eff, receives):
        """Deadline-monotonic response time, per core, solved by iteration."""
        on_core = defaultdict(list)
        for tid in self.task:
            on_core[alloc[tid]].append(tid)

        results = {}
        for core_id in self.core:
            cpu = self.core[core_id]["cpu_type"]
            klass = self.core[core_id]["core_class"]
            ov, tail = self.overhead[cpu], self.tail[klass]

            # receive work is an interrupt handler, so it sits above every task
            higher = [{"c": r["c"], "period": r["period"]} for r in receives.get(core_id, [])]

            for tid in sorted(on_core[core_id],
                              key=lambda i: priority_rank(self.task, i)):
                t = self.task[tid]
                ci, pi = c_eff[tid], t["period_us"]
                # the busy window opens with a wake, anything after it is a
                # dispatch onto a core that is already awake
                gap = ov["wake"] if not higher else ov["dispatch"]

                r, bounded = ov["wake"] + ci, True
                for _ in range(MAX_ITER):
                    # every activation ahead of this task costs the core that
                    # task's execution and its own dispatch
                    interference = sum(ceil(r / h["period"]) * (h["c"] + ov["dispatch"])
                                       for h in higher)
                    # an interferer's first release is simultaneous, so it
                    # blocks rather than preempts; every later one costs a resume
                    k = sum(max(0, ceil(r / h["period"]) - 1) for h in higher)
                    nxt = (ov["wake"] + ci + interference + k * ov["preempt"]) * tail
                    if abs(nxt - r) < 1e-9:
                        break
                    r = nxt
                    if r > t["deadline_us"]:
                        # past the deadline r can only grow, so the verdict is
                        # settled and r is a lower bound from here on
                        bounded = False
                        break

                results[tid] = {
                    "core": core_id, "cpu_type": cpu, "core_class": klass,
                    "c_eff": ci, "r": r, "deadline": t["deadline_us"],
                    "miss": r > t["deadline_us"], "r_bounded": bounded,
                    # own_cost is what this task alone costs the core per
                    # release: execution, the switch onto the core, and the
                    # resumes after preemption, all inflated by the class tail
                    "gap": gap, "preempts": k, "tail": tail,
                    "own_cost": (gap + ci + k * ov["preempt"]) * tail,
                }
                higher.append({"c": ci, "period": pi})

        return results

    def _chain(self, alloc, resp):
        """Worst path from src to dst. Every edge is sampled, so every hop
        costs a link delay, one consumer period of waiting, and the consumer's
        own response time."""
        def node_cost(n):
            return self.task[n]["period_us"] + resp[n]["r"]

        def edge_cost(a, c):
            return sum(lk["delay_us"] for lk in self.route(
                self.device_of(a, alloc), self.device_of(c, alloc)))

        out = []
        for b in self.budgets:
            found = worst_path(self.adj, b["src"], b["dst"], node_cost, edge_cost)
            if found is None:
                continue
            worst = resp[b["src"]]["r"] + found[0]
            worst_path_nodes = found[1]

            budget = b.get("budget_us")
            out.append({
                "src": b["src"], "dst": b["dst"], "budget_us": budget,
                "latency_us": worst, "path": worst_path_nodes,
                "violated": budget is not None and worst > budget,
                # a task on the path that missed has an unconverged r, so the
                # latency is a lower bound
                "exact": all(resp[n]["r_bounded"] for n in worst_path_nodes),
            })
        return out

    def _core_load(self, alloc, c_eff, receives, resp):
        """Two numbers per core.

        raw       sum of C_eff / period, execution only
        effective the same execution plus each task's dispatch gap and its
                  preemption switches, tail applied. Only knowable after
                  allocation, so the generator's load target cannot match it.
        """
        raw = {cid: 0.0 for cid in self.core}
        eff = {cid: 0.0 for cid in self.core}

        for tid in self.task:
            p = self.task[tid]["period_us"]
            raw[alloc[tid]] += c_eff[tid] / p
            eff[alloc[tid]] += resp[tid]["own_cost"] / p

        for cid, rs in receives.items():
            tail = self.tail[self.core[cid]["core_class"]]
            raw[cid] += sum(r["c"] / r["period"] for r in rs)
            eff[cid] += sum(r["c"] * tail / r["period"] for r in rs)

        return raw, eff

    def _penalty(self, alloc, resp, chains):
        misses = sum(
            self.task[tid]["internal_weight"] * EXTERNAL_WEIGHT[self.task[tid]["period_us"]]
            for tid, r in resp.items() if r["miss"]
        )
        violations = sum(
            max(self.task[n]["internal_weight"] * EXTERNAL_WEIGHT[self.task[n]["period_us"]]
                for n in c["path"])
            for c in chains if c["violated"]
        )
        preference = sum(
            preference_penalty(self.task[tid]["period_us"],
                               self.core[alloc[tid]]["core_class"])
            for tid in self.task
        )
        return {"misses": misses, "budgets": violations, "preference": preference,
                "total": misses + violations + preference}

    def evaluate(self, alloc):
        missing = set(self.task) - set(alloc)
        if missing:
            raise ValueError(f"unallocated tasks: {sorted(missing)}")

        c_eff, receives = self._effective_execution(alloc)
        resp = self._response_times(alloc, c_eff, receives)
        chains = self._chain(alloc, resp)
        raw, eff = self._core_load(alloc, c_eff, receives, resp)
        penalty = self._penalty(alloc, resp, chains)

        return {"tasks": resp, "cores": raw, "cores_effective": eff,
                "receives": dict(receives), "chains": chains, "penalty": penalty}
