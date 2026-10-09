"""Generate one task set, sized for a target platform and load.

    python scripts/generator/gen_taskset.py --platform 7 --coupling 2 --load 4 --n 31

Appends to the task set store, data/tasksets/tasksets.jsonl, as
taskset_<platform>_<coupling>_<load>_<n>. `--out` also dumps it as json.

Three axes, all set per instance:

  platform   which hardware it is sized for
  coupling   how much of the graph talks. Level 1 is loosely coupled, with many
             independent tasks the allocator can move for free. Level 4 has none.
  load       how full the cores are

Both coupling and load are jittered inside their cell, so the instances of a
cell label a small region rather than a point.

Shape is a main chain plus side branches, plus a set of tasks with no edges at
all. Every edge is sampled: the consumer reads on its own timer and never
waits, so periods never propagate.

Budgets come from the task graph alone, `floor x tightness`, so the same
workload is judged the same way whatever it runs on.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import tasksets
from allocators import REFERENCE, get
from timing import Model, worst_path

# --- the grid ---
# Coupling moves two things at once. Removing independent tasks alone changes
# little, because a chain plus branches is a tree and stays at about one edge
# per task however the level is set. The extra edges are what make a level
# denser, so a level 4 graph has real coupling to cut through. The axis
# saturates past about 0.4, above which the cells are simply hopeless.
COUPLING = {
    1: {"independent_frac": 0.35, "extra_edge_frac": 0.00},
    2: {"independent_frac": 0.25, "extra_edge_frac": 0.12},
    3: {"independent_frac": 0.15, "extra_edge_frac": 0.25},
    4: {"independent_frac": 0.05, "extra_edge_frac": 0.40},
}

# Levels target the EFFECTIVE load measured after allocation, not the estimate.
# A fixed estimate target means different things on different platforms, since
# bigger ones carry more crossings, receive work and preemption. Targeting the
# measured value makes a level mean the same thing everywhere.
#
# The band is a starting point, not a calibration: check it against the perfect
# rate and tie rate the analysis reports and move it if the levels do not
# separate.
LOAD = {1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1}
LOAD_TOLERANCE = 0.02         # how close the measured effective load must land
LOAD_MAX_ADJUST = 8           # give up rather than loop forever

# --- jitter, so a cell is a region rather than a point ---
LOAD_JITTER = 0.02
INDEP_JITTER = 0.04
EXTRA_EDGE_JITTER = 0.04
CHAIN_FRAC, CHAIN_JITTER = 0.60, 0.08

DEADLINE_RANGE = (0.8, 1.0)   # fraction of the period
WEIGHT_RANGE = (1.0, 2.0)     # internal_weight
BUDGET_TIGHTNESS = 1.8        # x the chain floor. 1.9 is on the line, 1.7 is tight.

MAX_TASKS = 600
ROOT = Path(__file__).resolve().parent.parent.parent


def draw(pool, n, rng):
    """n entries from pool, sampling without replacement until it runs out."""
    out = []
    while len(out) < n:
        out += rng.sample(pool, min(n - len(out), len(pool)))
    return out


def build(cat, count, rng, chain_frac, indep_frac, extra_edge_frac):
    groups = cat["groups"]
    by_role = {r: [c for c in cat["codes"] if c["role"] == r]
               for r in ("source", "mid", "sink")}
    gidx = {c["id"]: groups.index(c["group"]) for c in cat["codes"]}

    n_indep = min(count - 2, round(count * indep_frac)) if count > 2 else 0
    connected = count - n_indep
    chain_len = min(connected, max(2, round(connected * chain_frac)))

    chain = [rng.choice(by_role["source"])]
    mids = draw(by_role["mid"], max(0, chain_len - 2), rng)
    chain += sorted(mids, key=lambda c: (gidx[c["id"]], rng.random()))
    if chain_len >= 2:
        last = gidx[chain[-1]["id"]]
        tail = [c for c in by_role["sink"] if gidx[c["id"]] >= last] or by_role["sink"]
        chain.append(rng.choice(tail))

    branches = draw(by_role["mid"] + by_role["sink"], connected - len(chain), rng)
    loose = draw(cat["codes"], n_indep, rng)      # independence ignores role

    seen, ids = Counter(), []
    for c in chain + branches + loose:
        seen[c["id"]] += 1
        ids.append(c["id"] if seen[c["id"]] == 1 else f"{c['id']}_{seen[c['id']]}")
    chain_ids = ids[:len(chain)]
    branch_ids = ids[len(chain):len(chain) + len(branches)]

    edges = [{"src": a, "dst": b} for a, b in zip(chain_ids, chain_ids[1:])]
    attach = chain_ids[:-1] if len(chain_ids) > 1 else chain_ids

    # rank keeps the graph acyclic: an edge may only go from a lower rank to a
    # higher one. Chain nodes are ranked in order, a branch just after its parent.
    rank = {t: float(i) for i, t in enumerate(chain_ids)}
    for b in branch_ids:
        parent = rng.choice(attach)
        edges.append({"src": parent, "dst": b})
        rank[b] = rank[parent] + 0.5

    n_extra = round(extra_edge_frac * len(rank))
    if n_extra and len(rank) > 2:
        have = {(e["src"], e["dst"]) for e in edges}
        order = sorted(rank, key=rank.get)
        pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:]
                 if (a, b) not in have]
        rng.shuffle(pairs)
        edges += [{"src": a, "dst": b} for a, b in pairs[:n_extra]]

    tasks = []
    for tid, c in zip(ids, chain + branches + loose):
        p = c["period_us"]
        tasks.append({
            "task_id": tid,
            "code": c["code"],
            "period_us": p,
            "deadline_us": round(p * rng.uniform(*DEADLINE_RANGE)),
            "internal_weight": round(rng.uniform(*WEIGHT_RANGE), 2),
        })

    # floor is the worst simple path, matching the path the evaluator scores
    budgets = []
    if len(chain_ids) >= 2:
        adj = {}
        for e in edges:
            adj.setdefault(e["src"], []).append(e["dst"])
        period = {t["task_id"]: t["period_us"] for t in tasks}
        found = worst_path(adj, chain_ids[0], chain_ids[-1],
                           lambda n: period[n], lambda a, b: 0.0)
        floor = found[0] if found else period[chain_ids[-1]]
        budgets.append({"src": chain_ids[0], "dst": chain_ids[-1],
                        "budget_us": None, "floor_us": floor})

    return tasks, edges, budgets


def load_estimate(tasks, platform, exec_us, hw):
    """Mean core load, using only what is knowable before any allocation.

    Execution, one context switch per release, and the tail. Not preemption
    count and not link cpu cost: both depend on where things end up.
    """
    cores = platform["cores"]
    cpus = sorted({c["cpu_type"] for c in cores})
    mean_tail = sum(hw["core_class_tail"][c["core_class"]] for c in cores) / len(cores)
    mean_dispatch = sum(hw["cpu_overheads_us"][c]["dispatch"] for c in cpus) / len(cpus)

    total = 0.0
    for t in tasks:
        mean_c = sum(exec_us[t["code"]][c] for c in cpus) / len(cpus)
        total += (mean_c + mean_dispatch) / t["period_us"] * mean_tail
    return total / len(cores)


def seed_for(platform, coupling, load, n):
    return ((platform * 10 + coupling) * 10 + load) * 10000 + n


def generate(platform, coupling, load, n, tightness=BUDGET_TIGHTNESS,
             seed=None, out=None, quiet=False, cache=None, store=None):
    taskset_id = f"taskset_{platform}_{coupling}_{load}_{n}"
    platform_name = f"platform_{platform}"
    if seed is None:
        seed = seed_for(platform, coupling, load, n)
    rng = random.Random(seed)

    # jitter inside the cell
    cpl = COUPLING[coupling]
    load_target = max(0.05, LOAD[load] + rng.uniform(-LOAD_JITTER, LOAD_JITTER))
    indep_frac = min(0.6, max(0.0, cpl["independent_frac"]
                              + rng.uniform(-INDEP_JITTER, INDEP_JITTER)))
    extra_edge = max(0.0, cpl["extra_edge_frac"]
                     + rng.uniform(-EXTRA_EDGE_JITTER, EXTRA_EDGE_JITTER))
    chain_frac = min(0.9, max(0.3, CHAIN_FRAC
                              + rng.uniform(-CHAIN_JITTER, CHAIN_JITTER)))

    cache = cache if cache is not None else {}
    if "cat" not in cache:
        c = ROOT / "data" / "catalogues"
        cache["cat"] = json.loads((c / "task_catalogue.json").read_text())
        cache["exec"] = json.loads((c / "exec_table.json").read_text())["exec_us"]
        cache["hw"] = json.loads((c / "hardware_catalogue.json").read_text())
    if platform_name not in cache:
        cache[platform_name] = json.loads(
            (ROOT / "data" / "platforms" / f"{platform_name}.json").read_text())
    cat, exec_us, hw = cache["cat"], cache["exec"], cache["hw"]
    plat = cache[platform_name]

    def make(count):
        return build(cat, count, random.Random(seed), chain_frac, indep_frac, extra_edge)

    def measure(tasks, edges, budgets):
        """Effective load of the reference allocation: what the cores actually
        spend, including dispatch gaps and preemption switches."""
        probe = {"taskset_id": taskset_id, "tasks": tasks,
                 "edges": edges, "budgets": budgets}
        m = Model(probe, plat, exec_us, hw)
        res = m.evaluate(get(REFERENCE)(m, plat))
        eff = list(res["cores_effective"].values())
        return sum(eff) / len(eff), m, res

    # First guess from the cheap estimate, which knows nothing about preemption
    # or link cost and so undershoots. Then correct against the measured
    # effective load, which is what the level is defined in terms of.
    for count in range(2, MAX_TASKS):
        tasks, edges, budgets = make(count)
        est = load_estimate(tasks, plat, exec_us, hw)
        if est >= load_target * 0.8:
            break
    else:
        raise SystemExit(f"could not reach load {load_target} within {MAX_TASKS} tasks")

    # Effective load is only roughly monotone in task count, and on a small
    # platform one task moves it by 5%, so a proportional step oscillates.
    # Probe, then keep the closest probe rather than the last one.
    eff_mean, _, _ = measure(tasks, edges, budgets)
    best = (abs(eff_mean - load_target), count, tasks, edges, budgets, est, eff_mean)
    tried = {count}

    for _ in range(LOAD_MAX_ADJUST):
        if best[0] <= LOAD_TOLERANCE:
            break
        nxt = round(count * load_target / eff_mean) if eff_mean > 0 else count + 1
        nxt = max(3, min(MAX_TASKS - 1, nxt))
        while nxt in tried:                      # nudge towards the target
            nxt += 1 if eff_mean < load_target else -1
            nxt = max(3, min(MAX_TASKS - 1, nxt))
            if nxt in tried and abs(nxt - count) > 40:
                break
        if nxt in tried:
            break
        count = nxt
        tried.add(count)
        tasks, edges, budgets = make(count)
        est = load_estimate(tasks, plat, exec_us, hw)
        eff_mean, _, _ = measure(tasks, edges, budgets)
        if abs(eff_mean - load_target) < best[0]:
            best = (abs(eff_mean - load_target), count, tasks, edges, budgets, est, eff_mean)

    _, count, tasks, edges, budgets, est, eff_mean = best

    if budgets:
        budgets[0]["budget_us"] = round(budgets[0]["floor_us"] * tightness, 1)

    n_edges_per_task = len(edges) / len(tasks)
    inst = {
        "taskset_id": taskset_id,
        "generator": {
            "catalogue_id": cat["catalogue_id"],
            "platform_id": plat["platform_id"],
            "coupling_level": coupling,
            "load_level": load,
            "instance": n,
            "seed": seed,
            "count": len(tasks),
            "load_target_effective": round(load_target, 4),
            "load_estimated": round(est, 4),
            "chain_frac": round(chain_frac, 4),
            "independent_frac": round(indep_frac, 4),
            "extra_edge_frac": round(extra_edge, 4),
            "n_independent": sum(1 for t in tasks if t["task_id"] not in
                                 {e["src"] for e in edges} | {e["dst"] for e in edges}),
            "edges_per_task": round(n_edges_per_task, 4),
            "deadline_range": list(DEADLINE_RANGE),
            "weight_range": list(WEIGHT_RANGE),
            "budget_tightness": tightness,
            "budget_rule": "chain floor x budget_tightness, no platform involved",
            "edge_semantics": "sampled, all edges",
        },
        "tasks": tasks,
        "edges": edges,
        "budgets": budgets,
    }

    # the reference allocation was already computed while sizing, so reuse it
    model = Model(inst, plat, exec_us, hw)
    res = model.evaluate(get(REFERENCE)(model, plat))
    raw = list(res["cores"].values())
    eff = list(res["cores_effective"].values())
    inst["generator"].update({
        "reference": REFERENCE,
        "reference_penalty": round(res["penalty"]["total"], 3),
        "reference_chain_us": round(res["chains"][0]["latency_us"], 1) if budgets else None,
        "load_actual_mean": round(sum(raw) / len(raw), 4),
        "load_actual_max": round(max(raw), 4),
        "load_effective_mean": round(sum(eff) / len(eff), 4),
        "load_effective_max": round(max(eff), 4),
    })

    # storage is the caller's business: `gen_dataset` keeps one writer open for
    # the whole grid, and a caller that only wants the numbers passes neither
    if store is not None:
        store.write(inst)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(inst, indent=2) + "\n")

    if not quiet:
        g = inst["generator"]
        print(f"{taskset_id:<28} {platform_name:<12} {len(tasks):>3} tasks  "
              f"indep {g['n_independent']:>2}  "
              f"load target {g['load_target_effective']:.2f}  "
              f"eff {g['load_effective_mean']:.2f}  "
              f"ref penalty {g['reference_penalty']}")
    return inst


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", type=int, required=True)
    ap.add_argument("--coupling", type=int, required=True, choices=sorted(COUPLING))
    ap.add_argument("--load", type=int, required=True, choices=sorted(LOAD))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tightness", type=float, default=BUDGET_TIGHTNESS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="dump this one task set to a json file as well")
    ap.add_argument("--no-store", action="store_true",
                    help="print it without adding it to the store")
    args = ap.parse_args()
    store = None if args.no_store else tasksets.Writer()
    try:
        generate(args.platform, args.coupling, args.load, args.n,
                 tightness=args.tightness, seed=args.seed, out=args.out,
                 store=store)
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    main()
