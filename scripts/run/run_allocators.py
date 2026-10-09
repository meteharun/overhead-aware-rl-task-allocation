"""Run every allocator over every task set and save the allocations.

    python scripts/run/run_allocators.py                     all task sets, all allocators
    python scripts/run/run_allocators.py --only bfd wfd      just the one pass packers
    python scripts/run/run_allocators.py --only sa_bfd sa_wfd --per-cell 10
    python scripts/run/run_allocators.py --platforms 3 4

Writes data/allocations/<allocator>.jsonl, one line per task set, holding the
allocation plus a summary of how it scored. The allocation is the source of
truth; the summary can always be recomputed through `timing.Model`.

The annealed allocators are three orders of magnitude slower than the one pass
ones, because they call the timing model a few thousand times instead of once.
`--per-cell K` keeps only the first K instances of every (platform, coupling,
load) cell, which runs them over the whole grid without the whole dataset.
Pools of different size are fine: the analysis pairs on task set id, so it
compares only what both allocators saw.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import splits
import tasksets
from allocators import configure_exact, configure_sa, get, learned, names
from progress import Progress
from timing import EXTERNAL_WEIGHT, Model

ROOT = Path(__file__).resolve().parent.parent.parent
PLATFORMS = ROOT / "data" / "platforms"
OUT = ROOT / "data" / "allocations"


def summarise(model, alloc, res, extra=None, ms=None):
    misses = [t for t, r in res["tasks"].items() if r["miss"]]
    loads = list(res["cores"].values())
    eff = list(res["cores_effective"].values())
    chain = res["chains"][0] if res["chains"] else None
    p = res["penalty"]

    return {
        # wall clock of the allocator call only, in milliseconds. Building the
        # Model and scoring the result afterwards are the harness and are not
        # charged to it.
        "alloc_ms": round(ms, 4) if ms is not None else None,

        # search allocators report the penalty of the allocation they were
        # handed and how far they moved it
        **(extra or {}),
        "penalty_total": round(p["total"], 4),
        "penalty_misses": round(p["misses"], 4),
        "penalty_budgets": round(p["budgets"], 4),
        "penalty_preference": round(p["preference"], 4),
        # misses plus budgets, without the preference tiebreaker. Comparing on
        # total would let preference decide instances where nothing failed.
        "penalty_hard": round(p["misses"] + p["budgets"], 4),

        "n_tasks": len(model.task),
        "n_missed": len(misses),
        "missed_hard": sum(1 for t in misses
                           if EXTERNAL_WEIGHT[model.task[t]["period_us"]] == 100),
        "missed_firm": sum(1 for t in misses
                           if EXTERNAL_WEIGHT[model.task[t]["period_us"]] == 10),
        "missed_soft": sum(1 for t in misses
                           if EXTERNAL_WEIGHT[model.task[t]["period_us"]] == 1),
        "perfect": len(misses) == 0 and not (chain and chain["violated"]),

        "chain_latency_us": round(chain["latency_us"], 1) if chain else None,
        "chain_budget_us": chain["budget_us"] if chain else None,
        "chain_violated": bool(chain and chain["violated"]),

        "n_crossings": sum(1 for a, b in model.edges
                           if model.device_of(a, alloc) != model.device_of(b, alloc)),
        "n_edges": len(model.edges),
        "n_devices_used": len({model.core[c]["device_id"] for c in alloc.values()}),
        "cores_used": len(set(alloc.values())),

        "load_mean": round(sum(loads) / len(loads), 4),
        "load_max": round(max(loads), 4),
        "load_effective_mean": round(sum(eff) / len(eff), 4),
        "load_effective_max": round(max(eff), 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=names(), nargs="+",
                    help="one or more allocator names. default: all of them")
    ap.add_argument("--platforms", type=int, nargs="*", default=None)
    ap.add_argument("--per-cell", type=int, default=None, metavar="K",
                    help="keep only the first K instances of each cell. "
                         "Use it to run the annealed allocators over the whole "
                         "grid without running them over the whole dataset")
    ap.add_argument("--eval", action="store_true",
                    help="only task sets the learned policy never trained on. "
                         "Any headline number goes here")
    ap.add_argument("--exact-seconds", type=float, default=None, metavar="S",
                    help="time limit per instance for the exact allocator. "
                         "It is exponential, so this is the whole cost control: "
                         "instances it cannot finish come back with the best "
                         "allocation found and say so")
    ap.add_argument("--sa-evals", type=int, default=None, metavar="N",
                    help="candidate evaluations for every annealed allocator. "
                         "Default 4000, which is what the thesis compares "
                         "starting points at. A larger budget asks instead "
                         "whether an instance is solvable at all. The "
                         "checkpoint ladder extends with it, so the budget "
                         "curve keeps its earlier points and gains new ones")
    ap.add_argument("--append", action="store_true",
                    help="add to the existing files instead of replacing them, "
                         "skipping task sets already in there, to extend the "
                         "results to new platforms without redoing the "
                         "annealing")
    args = ap.parse_args()

    if args.sa_evals is not None:
        if args.sa_evals < 1:
            raise SystemExit("--sa-evals must be at least 1")
        configure_sa(args.sa_evals)
    allocators = args.only if args.only else names()
    if args.exact_seconds is not None:
        configure_exact(time_limit=args.exact_seconds)
    # Instances are numbered from 1 inside each cell, so --per-cell is a
    # stratified sample by construction, with no cell left out.
    wanted = tasksets.select(platforms=args.platforms, per_cell=args.per_cell,
                             keep=splits.is_eval if args.eval else None)
    if not wanted:
        raise SystemExit("no task sets found. run scripts/generator/gen_dataset.py first")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(wanted)} task sets x {len(allocators)} allocators "
          f"= {len(wanted) * len(allocators)} allocations")
    print(f"allocators: {', '.join(allocators)}"
          + ("   (eval only, nothing the policy trained on)" if args.eval else ""))
    if any(a.startswith("sa_") for a in allocators) and not args.per_cell:
        print("note: an annealed allocator costs a few thousand timing model "
              "calls per task set. --per-cell K cuts this down.")
    print()

    bar = Progress(len(wanted), "task sets")

    # appending is idempotent: whatever is already recorded is skipped, so
    # rerunning the same command cannot double up rows
    already = {a: set() for a in allocators}
    if args.append:
        for a in allocators:
            f = OUT / f"{a}.jsonl"
            if f.exists():
                with f.open() as fh:
                    already[a] = {json.loads(line)["taskset_id"] for line in fh}
        have = sum(len(v) for v in already.values())
        todo = sum(1 for t in wanted for a in allocators if t not in already[a])
        print(f"appending: {have} allocations already on disk, {todo} to compute\n")

    # Reading a checkpoint inside the first timed allocation would put the file
    # read and the torch import into that instance's alloc_ms, so it happens
    # here instead. `exact` warm starts from the policy, so it needs this too.
    needs = {a.removeprefix("sa_") for a in allocators} | (
        {"rl"} if "exact" in allocators else set())
    if needs & set(learned.trained()):
        learned.preload(sorted(needs & set(learned.trained())))

    platforms = {}
    handles = {a: (OUT / f"{a}.jsonl").open("a" if args.append else "w")
               for a in allocators}
    try:
        for i, (tid, taskset) in enumerate(tasksets.stream(wanted), start=1):
            g = taskset["generator"]
            pid = g["platform_id"]
            if pid not in platforms:
                platforms[pid] = json.loads((PLATFORMS / f"{pid}.json").read_text())
            platform = platforms[pid]

            for name in allocators:
                if taskset["taskset_id"] in already[name]:
                    continue
                model = Model(taskset, platform)
                # so an allocator that wants the instance's saved results can
                # find them. `exact` opens from the best allocation on file
                model.taskset_id = taskset["taskset_id"]
                fn = get(name)
                t_alloc = time.perf_counter()
                alloc = fn(model, platform)
                ms = 1000 * (time.perf_counter() - t_alloc)
                # search allocators leave their statistics on the function. A
                # one pass allocator has no `last`, so this is None for them.
                extra = getattr(fn, "last", None)
                res = model.evaluate(alloc)
                handles[name].write(json.dumps({
                    "taskset_id": taskset["taskset_id"],
                    "set": splits.set_of(taskset["taskset_id"]),
                    "platform_id": pid,
                    "allocator": name,
                    "coupling_level": g["coupling_level"],
                    "load_level": g["load_level"],
                    "n_independent": g["n_independent"],
                    "edges_per_task": g["edges_per_task"],
                    "allocation": alloc,
                    "summary": summarise(model, alloc, res, extra, ms),
                }) + "\n")

            bar.tick(i)
    finally:
        for h in handles.values():
            h.close()

    for a in allocators:
        print(f"{a:<10} -> data/allocations/{a}.jsonl   ({len(wanted)} rows)")
    bar.close()
    print("next: python scripts/analysis/analyse_allocations.py")


if __name__ == "__main__":
    main()
