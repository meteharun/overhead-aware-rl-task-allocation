#!/usr/bin/env python3
"""Repair, then the same search budget a rebuild gets.

    python scripts/run/repair_search.py                    the full pool, about four hours
    python scripts/run/repair_search.py --n 400            a sample first
    python scripts/run/repair_search.py --append           carry on where it stopped

Writes out/runs/repair_search.jsonl, one row per failure per allocator, each
carrying the budget curve of the search that followed the repair.

`repair.py` gave every reaction ten evaluations of follow-up search and gave a
rebuild a thousand. That is the right comparison for a failure window and the
wrong one for the question of whether repairing is a good place to start from,
because the two were never given the same budget. This closes that: take the
orphans-only repair and anneal it exactly as a rebuild is annealed, same move
set, same schedule, same thousand evaluations, recorded at the same checkpoints.
The only difference left between the two curves is where the search began, which
is the same comparison the offline chapter makes between starting points.

The search is unrestricted, as the rebuild's is. It may move survivors that were
pinned during construction, so a repair at a thousand evaluations is no longer
orphans-only and its migration count climbs. That number is recorded at every
checkpoint, because it is the price of the quality and the whole reason repair
was attractive in the first place.

Replay, not resampling. The pool, the seed and the order of `victims` calls are
the ones `repair.py` uses, so this hits the same task sets and the same dead
cores as the run already on disk and the two files can be read side by side.
Skipping a task set with `--append` still draws its victims, or the stream would
diverge from that point on.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import splits
import tasksets
from allocators import can_repair, get, repair as repair_with
from allocators.sa import anneal, seed_for
from progress import Progress, hms
from repair import hard, read_base, victims, without
from timing import Model

PLATFORMS = ROOT / "data" / "platforms"
OUT = ROOT / "out" / "runs" / "repair_search.jsonl"
MS_PER_EVAL = 0.30          # the estimate repair.py uses, for the same warning


def done_keys(path):
    """(task set, dead core, allocator) already on disk."""
    keys = set()
    if not path.exists():
        return keys
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a line cut in half by ^C
            keys.add((r["taskset"], r["dead_core"], r["allocator"]))
    return keys


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="sa_rl",
                    help="the running allocation that the failure hits")
    ap.add_argument("--allocators", nargs="+",
                    default=["bfd", "wfd", "heft", "rl"])
    ap.add_argument("--failures", type=int, default=2,
                    help="dead cores per task set, one at a time")
    ap.add_argument("--iters", type=int, default=1000,
                    help="evaluations of search after the repair. The default "
                         "is what a rebuild is given, which is the point")
    ap.add_argument("--n", type=int, default=None,
                    help="only this many task sets, spread over the pool. This "
                         "breaks the replay: a shorter pool draws a different "
                         "sequence of victims, so the failures will not be the "
                         "ones in repair.jsonl. Use it to time the run, not to "
                         "produce results")
    ap.add_argument("--seed", type=int, default=0,
                    help="must match the seed repair.py ran with, or the "
                         "failures will not be the same ones")
    ap.add_argument("--append", action="store_true",
                    help="skip failures already in the output and keep going")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    for a in args.allocators:
        if not can_repair(a):
            raise SystemExit(f"{a} has no repair mode. Drop it.")

    base = read_base(args.base)
    pool = [t for t in tasksets.select(keep=splits.is_eval) if t in base]
    if not pool:
        raise SystemExit(f"{args.base} has no eval allocations on file")
    if args.n and args.n < len(pool):
        pool = pool[::len(pool) // args.n][:args.n]

    out = Path(args.out)
    already = done_keys(out) if args.append else set()
    n_fail = len(pool) * args.failures
    todo = n_fail * len(args.allocators) - len(already)

    print(f"{len(pool)} task sets x {args.failures} failures = {n_fail} failures")
    print(f"running allocation: {args.base}")
    print(f"repairing with: {', '.join(args.allocators)}")
    print(f"then annealing each repair for {args.iters} evaluations")
    if already:
        print(f"appending: {len(already):,} already on disk")
    print(f"\nestimate: {todo:,} searches at about "
          f"{args.iters * MS_PER_EVAL / 1000:.1f} s each = "
          f"{hms(todo * args.iters * MS_PER_EVAL / 1000)}\n")

    rng = random.Random(args.seed)
    cache = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    bar = Progress(len(pool), "task sets")
    rows = 0
    stopped = False

    with out.open("a" if args.append else "w") as fh:
        try:
            for i, (tid, taskset) in enumerate(tasksets.stream(pool), start=1):
                pid = taskset["generator"]["platform_id"]
                if pid not in cache:
                    cache[pid] = json.loads(
                        (PLATFORMS / f"{pid}.json").read_text())
                platform = cache[pid]
                running = base[tid]

                # drawn for every task set, finished or not, so that skipping
                # one does not shift every later draw
                dead_cores = victims(platform, running, args.failures, rng)
                if all((tid, d, a) in already
                       for d in dead_cores for a in args.allocators):
                    bar.tick(i, f"{rows:,} rows")
                    continue

                before = hard(Model(taskset, platform), running)
                g = taskset["generator"]
                for dead in dead_cores:
                    reduced = without(platform, dead)
                    model = Model(taskset, reduced)
                    model.taskset_id = tid
                    survivors = {t: c for t, c in running.items() if c != dead}

                    def moved_from(a):
                        return sum(1 for t, c in a.items() if running[t] != c)

                    for name in args.allocators:
                        if (tid, dead, name) in already:
                            continue
                        t0 = time.perf_counter()
                        alloc = repair_with(name, model, reduced, survivors)
                        build_ms = (time.perf_counter() - t0) * 1000
                        best, st = anneal(
                            model, alloc, iters=args.iters, stall=None,
                            rng=random.Random(seed_for(model)),
                            track=lambda a: {"moved": moved_from(a)})
                        fh.write(json.dumps({
                            "taskset": tid, "platform": pid,
                            "dead_core": dead, "allocator": name,
                            "freed": "0",
                            "orphans": len(running) - len(survivors),
                            "n_tasks": len(model.task),
                            "load": g["load_level"],
                            "coupling": g["coupling_level"],
                            "penalty_before": before,
                            "penalty_after": hard(model, alloc),
                            "migrations": moved_from(alloc),
                            "ms": round(build_ms, 4),
                            "searched_penalty": hard(model, best),
                            "searched_migrations": moved_from(best),
                            "sa_curve": st["sa_curve"],
                        }) + "\n")
                        rows += 1
                # one flush per task set rather than per row, which is cheap
                # next to a thousand evaluations and keeps ^C from costing more
                # than the task set in flight
                fh.flush()
                bar.tick(i, f"{rows:,} rows")
        except KeyboardInterrupt:
            stopped = True

    bar.close()
    if stopped:
        print(f"\nstopped by ^C after {rows:,} rows. Rerun the same command "
              f"with --append to carry on.")
    print(f"wrote {rows:,} rows to {out}")


if __name__ == "__main__":
    main()
