"""A core dies. What happens next.

    python scripts/run/repair.py
    python scripts/run/repair.py --n 200                  a pilot
    python scripts/run/repair.py --rebuild-one-pass       the cheap contrast

Writes out/runs/repair.jsonl and prints the tables.

Every task set in the eval split, two failures each, one core at a time. The
cores are drawn from the non-os cores that actually hold work, so every failure
strands something and no row is a core nobody was using. os cores are never
failed: every device must have exactly one, it is where all incoming link
traffic lands, and losing it loses the whole device rather than one core, which
is a different experiment.

**What was running when it died** comes off disk, from `--base`, default
`sa_rl`. That is the offline conclusion of the rest of the project, so it is the
allocation a deployed system would be running, and the same starting point for
every allocator here. Recomputing a one pass `rl` allocation instead would make
re-optimising after the failure score better than before it, because "before"
would never have been optimised at all.

**The spectrum, not two options.** Keeping every survivor pinned moves the
fewest tasks and gives up quality, because it has to work around a placement
decided before the core died. Rebuilding re-runs the offline pipeline and may
move everything. Neither is obviously right, because the objective charges
nothing for a migration and a running system charges a great deal. So the axis
is swept: alongside the orphans, free the K least critical survivors and let the
allocator re-place those too.

  K = 0      only the orphans move. Full repair.
  K = 1..32  that many survivors are freed as well.
  K = all    everything is re-placed. Full rebuild.

Least critical first, on purpose: those are the tasks the objective cares least
about losing, so they are the ones worth disturbing to make room for an orphan
that matters.

**A rebuild gets the search.** It is the offline pipeline run again on a
platform with one fewer core, so it runs the annealed allocator rather than the
one pass one. The allocation that was running is annealed, and comparing it
against a one pass rebuild only shows that one pass allocations are worse than
annealed ones, which the offline results already said. `--rebuild-one-pass`
restores the cheap version if the contrast is wanted.

**The budget stops at 1,000 evaluations, not the offline 4,000.** This is a
dynamic allocation experiment and a third of a second is already at the edge of
what a failure window can spend. Measuring out to 4,000 would be pricing a
reaction nobody could deploy, and the crossover the experiment exists to find,
the budget at which rebuilding beats patching the hole, sits nearer 250. The
best so far is written down at every budget on the way, on the same marks the
offline search experiment uses, so the two read against each other directly.

That still makes the last rung a different kind of thing from the ones before
it, spending hundreds of milliseconds where they spend single digits. It is
labelled separately in the tables and the curve should not be read as continuous
across it.

**Every reaction also gets a short search**, `--search` evaluations of
annealing, default 10. That is a failure window budget, not a search budget. The
same number for every reaction, so the only thing separating the results is
where the search started.

**Two numbers per reaction**, and the second is the one nothing else measures:

  penalty     how good the result is
  migrations  how many tasks had to stop and restart somewhere else. Nothing in
              the objective charges for this, and in a running system it is the
              entire cost of reacting.
"""

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import splits
import tasksets
from allocators import can_repair, get, repair as repair_with
from allocators.sa import anneal, seed_for
from progress import Progress, hms
from timing import EXTERNAL_WEIGHT, Model

ROOT = Path(__file__).resolve().parent.parent.parent
PLATFORMS = ROOT / "data" / "platforms"
ALLOC = ROOT / "data" / "allocations"
OUT = ROOT / "out" / "runs"

DEFAULT = ["rl", "bfd", "wfd", "heft"]
BUDGETS = [0, 1, 2, 4, 8, 16, 32]      # survivors freed on top of the orphans


def without(platform, core_id):
    """The same platform with one core gone."""
    return {**platform,
            "cores": [c for c in platform["cores"] if c["core_id"] != core_id]}


def hard(model, alloc):
    p = model.evaluate(alloc)["penalty"]
    return p["misses"] + p["budgets"]


def read_base(name):
    """The allocation that was running, per task set."""
    f = ALLOC / f"{name}.jsonl"
    if not f.exists():
        raise SystemExit(f"no {f}. run: python scripts/run/run_allocators.py "
                         f"--only {name} --eval")
    out = {}
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["taskset_id"]] = r["allocation"]
    return out


def victims(platform, alloc, k, rng):
    """k non-os cores that are carrying work, drawn without replacement."""
    busy = set(alloc.values())
    pool = [c["core_id"] for c in platform["cores"]
            if c["core_class"] != "os" and c["core_id"] in busy]
    rng.shuffle(pool)
    return pool[:k]


def cheapest_first(model, survivors):
    """Survivors ordered by what missing them would cost, least first.

    Ties by task id so the same K frees the same tasks on every run.
    """
    def weight(t):
        task = model.task[t]
        return (task["internal_weight"] * EXTERNAL_WEIGHT[task["period_us"]], t)

    return sorted(survivors, key=weight)


def label(k):
    return "rebuild" if k is None else str(k)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="sa_rl",
                    help="the allocator whose saved allocation was running when "
                         "the core died")
    ap.add_argument("--allocators", nargs="+", default=DEFAULT,
                    help="which allocators react, across the whole spectrum")
    ap.add_argument("--failures", type=int, default=2,
                    help="cores to fail per task set, one at a time")
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGETS,
                    help="survivors freed on top of the orphans. A full rebuild "
                         "is always added on the end")
    ap.add_argument("--search", type=int, default=10, metavar="EVALS",
                    help="annealing evaluations allowed after reacting. A "
                         "failure window budget, not a search budget. 0 to skip")
    ap.add_argument("--rebuild-one-pass", action="store_true",
                    help="rebuild with the plain allocator instead of its "
                         "annealed twin. Much faster and much worse, and it "
                         "compares a one pass rebuild against an annealed "
                         "allocation, which is not a fair question")
    ap.add_argument("--rebuild-iters", type=int, default=1000,
                    help="search budget for a rebuild. 1000 evaluations is "
                         "about a third of a second, which is already at the "
                         "edge of what a failure window can spend. 4000 is the "
                         "offline budget and is worth running on a sample, as "
                         "a ceiling rather than as an option")
    ap.add_argument("--n", type=int, default=None,
                    help="only this many task sets, spread over the pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "repair.jsonl"))
    args = ap.parse_args()

    for a in args.allocators:
        if not can_repair(a):
            raise SystemExit(f"{a} has no repair mode, so it cannot be compared "
                             "on one. Drop it from --allocators.")
        if not args.rebuild_one_pass:
            get(f"sa_{a}")          # fails now rather than an hour in

    base = read_base(args.base)
    pool = [t for t in tasksets.select(keep=splits.is_eval) if t in base]
    if not pool:
        raise SystemExit(f"{args.base} has no eval allocations on file")
    if args.n and args.n < len(pool):
        pool = pool[::len(pool) // args.n][:args.n]

    ladder = sorted(set(args.budgets)) + [None]     # None is the full rebuild
    per_failure = len(args.allocators) * len(ladder)
    n_fail = len(pool) * args.failures

    print(f"{len(pool)} task sets x {args.failures} failures = {n_fail} failures")
    print(f"running allocation: {args.base}")
    print(f"reacting with: {', '.join(args.allocators)}")
    print(f"freed survivors: {', '.join(label(k) for k in ladder)}")
    print("rebuild: " + ("one pass" if args.rebuild_one_pass else
                         f"annealed, {args.rebuild_iters} evaluations, "
                         "recorded at every budget on the way"))
    print(f"search after each: {args.search} evaluations"
          if args.search else "no search")
    print(f"{per_failure} reactions per failure, {n_fail * per_failure} rows")
    if not args.rebuild_one_pass:
        # the rebuilds dominate everything else by three orders of magnitude,
        # so the estimate is just them
        secs = n_fail * len(args.allocators) * args.rebuild_iters * 0.30 / 1000
        print(f"\nestimate: {n_fail * len(args.allocators):,} annealed rebuilds "
              f"at about {args.rebuild_iters * 0.30 / 1000:.1f} s each "
              f"= {hms(secs)}")
        print("  everything else in the ladder is milliseconds and rounds to "
              "nothing")
    print()

    rng = random.Random(args.seed)
    cache = {}
    stat = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    bar = Progress(len(pool), "task sets")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")
    try:
        for i, (tid, taskset) in enumerate(tasksets.stream(pool), start=1):
            pid = taskset["generator"]["platform_id"]
            if pid not in cache:
                cache[pid] = json.loads((PLATFORMS / f"{pid}.json").read_text())
            platform = cache[pid]

            running = base[tid]
            before = hard(Model(taskset, platform), running)

            for dead in victims(platform, running, args.failures, rng):
                reduced = without(platform, dead)
                model = Model(taskset, reduced)
                model.taskset_id = tid
                survivors = {t: c for t, c in running.items() if c != dead}
                orphans = len(running) - len(survivors)
                order = cheapest_first(model, survivors)

                moved_from = lambda a: sum(1 for t, c in a.items()
                                           if running[t] != c)
                jobs = [(name, k) for name in args.allocators for k in ladder]
                for name, k in jobs:
                    curve = None
                    if k is None and not args.rebuild_one_pass:
                        # the offline pipeline, run again on the reduced
                        # platform. Called apart rather than through the sa_
                        # registry entry so the budget curve can carry
                        # migrations, which needs the allocation at each
                        # checkpoint and not only its score
                        t0 = time.perf_counter()
                        start = get(name)(model, reduced)
                        alloc, st = anneal(
                            model, start, iters=args.rebuild_iters, stall=None,
                            rng=random.Random(seed_for(model)),
                            track=lambda a: {"moved": moved_from(a)})
                        ms = (time.perf_counter() - t0) * 1000
                        curve = st["sa_curve"]
                    elif k is None:
                        t0 = time.perf_counter()
                        alloc = get(name)(model, reduced)
                        ms = (time.perf_counter() - t0) * 1000
                    else:
                        freed = set(order[:k])
                        pinned = {t: c for t, c in survivors.items()
                                  if t not in freed}
                        t0 = time.perf_counter()
                        alloc = repair_with(name, model, reduced, pinned)
                        ms = (time.perf_counter() - t0) * 1000

                    after = hard(model, alloc)
                    moved = moved_from(alloc)

                    s_after, s_moved, s_ms = after, moved, ms
                    if args.search:
                        t1 = time.perf_counter()
                        best, _ = anneal(model, alloc, iters=args.search,
                                         stall=None, checkpoints=(),
                                         rng=random.Random(seed_for(model)))
                        s_ms = ms + (time.perf_counter() - t1) * 1000
                        s_after = hard(model, best)
                        s_moved = sum(1 for t, c in best.items()
                                      if running[t] != c)

                    how = f"{name}:{label(k)}"
                    v = stat[how]
                    v["penalty"].append(after)
                    v["delta"].append(after - before)
                    v["migrations"].append(moved)
                    v["ms"].append(ms)
                    v["s_penalty"].append(s_after)
                    v["s_migrations"].append(s_moved)
                    v["s_ms"].append(s_ms)
                    v["orphans"].append(orphans)
                    v["tasks"].append(len(running))
                    v["before"].append(before)

                    fh.write(json.dumps({
                        "taskset": tid, "platform": pid, "dead_core": dead,
                        "allocator": name, "freed": label(k),
                        "orphans": orphans, "n_tasks": len(running),
                        "load": taskset["generator"]["load_level"],
                        "coupling": taskset["generator"]["coupling_level"],
                        "penalty_before": round(before, 4),
                        "penalty_after": round(after, 4),
                        "migrations": moved, "ms": round(ms, 4),
                        "searched_penalty": round(s_after, 4),
                        "searched_migrations": s_moved,
                        "searched_ms": round(s_ms, 4),
                        # only the rebuild rung has one: what it had at 10, 25,
                        # 50 ... evaluations, on the same budgets the offline
                        # search experiment uses
                        **({"sa_curve": curve} if curve else {})}) + "\n")
                    n_rows += 1

            bar.tick(i, f"{n_rows} rows")
    finally:
        fh.close()

    report(stat, args, ladder)
    bar.close()
    print(f"wrote {out}  ({n_rows} rows)")


def report(stat, args, ladder):
    order = [f"{n}:{label(k)}" for n in args.allocators for k in ladder]
    order = [h for h in order if h in stat] + \
            [h for h in stat if h not in set(order)]
    first = stat[order[0]]
    n_fail = len(first["penalty"])
    tasks = statistics.mean(first["tasks"])
    orph = statistics.mean(first["orphans"])

    print(f"\n{'=' * 108}\n{n_fail} single core failures, "
          f"{orph:.1f} orphans of {tasks:.0f} tasks on average\n{'=' * 108}")
    print("left of the bar is the reaction on its own, right of it is the same "
          f"reaction plus {args.search} evaluations of search\n")
    print(f"{'reaction':<18}{'penalty':>10}{'moved':>8}{'% tasks':>9}"
          f"{'med ms':>9}  |{'penalty':>10}{'moved':>8}{'med ms':>9}")
    last = None
    for how in order:
        v = stat[how]
        name = how.split(":")[0]
        if last is not None and name != last:
            print()
        last = name
        mig = statistics.mean(v["migrations"])
        print(f"{how:<18}{statistics.mean(v['penalty']):>10.1f}{mig:>8.1f}"
              f"{100 * mig / tasks:>8.0f}%"
              f"{statistics.median(v['ms']):>9.2f}  |"
              f"{statistics.mean(v['s_penalty']):>10.1f}"
              f"{statistics.mean(v['s_migrations']):>8.1f}"
              f"{statistics.median(v['s_ms']):>9.2f}")

    print(f"\n  penalty before the failure: "
          f"{statistics.mean(first['before']):.1f}   (the {args.base} allocation "
          "that was running)")
    print("  migrations counts every task whose core changed, orphans included, "
          "so K=0 can never be below the orphan count")
    print(f"  every reaction saw the same {n_fail} failures, so the columns are "
          "paired")


if __name__ == "__main__":
    main()
