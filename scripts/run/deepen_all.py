#!/usr/bin/env python3
"""Keep annealing every allocator's unsolved instances, to a fixed step budget.

    python scripts/run/deepen_all.py                       all starts, 500k steps each
    python scripts/run/deepen_all.py --only sa_bfd sa_wfd   one or two at a time
    python scripts/run/deepen_all.py --evals 100000         a shorter pass first
    python scripts/run/deepen_all.py --n 100                a sample, spread over the pool

Rerun the same command to carry on. The script always resumes: it skips the
(allocator, instance) pairs it has already finished and picks a half-finished
search up from its last completed round, so ^C or a power cut costs at most one
round.

Writes out/runs/deepen_all.jsonl, one row per allocator per instance, and
out/runs/deepen_all.state.jsonl, the per-round state it resumes from. Both are
append-only and the last row for a pair wins.

What this is for
----------------
The thesis stops every search at 4,000 candidate evaluations and compares what
each starting point bought by then. This asks the other question: given a budget
large enough to matter, which starts still cannot finish the job, and on which
instances. Every allocator gets the same budget on the same instances, so the
comparison stays like for like.

The budget is counted in steps, not seconds. A time budget gives a small
instance six million evaluations and a large one three hundred thousand, which
is fine for asking whether an instance is feasible and useless for comparing
allocators.

The recorder sits in the model wrapper rather than in `anneal`, so it sees every
evaluation and snapshots at a fixed ladder of budgets whatever the rounds do.

It stops on the first perfect allocation, whatever the preference penalty. The
run is looking for a zero-timing-penalty witness, so that is the only stopping
test: no deadline miss, no chain overrun. The moment the search evaluates such
an allocation it is thrown out of the anneal and the pair is finished, mid
round, rather than spending the rest of its budget tidying up the core-class
fit.

The search objective still keeps preference, demoted to a tie-break. `sa.py`
minimises `penalty.total`, which includes preference, and that is right for the
thesis comparison but wrong as a stopping test, since a soft miss costs as
little as 1.0 while preference reaches 0.5 per task. The wrapper redefines the
number the search reads as

    misses + budget overruns + PREF_WEIGHT x preference

which puts nearly all the weight on the timing penalty while keeping the small
gradient `sa.py` relies on across plateaus where nothing misses. Nothing in
timing.py or sa.py is changed.

The budget is spent in rounds, each a full anneal from the best allocation so
far with the temperature rescaled to its penalty. One cooling schedule stretched
over 500,000 evaluations spends almost all of it at a temperature that accepts
nothing, and a round boundary is a natural checkpoint to disk. This restart
annealing is a deliberate departure from the single 4,000-evaluation schedule
the thesis compares on, and has to be described as such if these numbers are
used.
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
from allocators.sa import anneal, seed_for
from progress import Progress
from timing import Model, load_catalogues

ALLOC = ROOT / "data" / "allocations"
PLATFORMS = ROOT / "data" / "platforms"
OUT = ROOT / "out" / "runs" / "deepen_all.jsonl"

BASE_BUDGET = 4000        # what the thesis run already spent, before any of this
EVALS = 500_000           # extra evaluations per instance
ROUND_EVALS = 50_000      # one reheat, so ten of them fill the default budget
PREF_WEIGHT = 0.01        # preference demoted to a tie-break
MIN_MISS = 1.0            # cheapest possible miss: internal weight 1.0 x external 1
MAX_PREF_PER_TASK = 0.5   # a HARD task on an os core, the worst entry in Table 3.4

# Budgets to snapshot at, counted in evaluations past the 4,000 already spent.
# Geometric, so the log axis the budget figure uses gets even spacing, and dense
# at the low end because that is where most of the progress happens.
LADDER = (100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000,
          100_000, 250_000, 500_000)

# sa_rl_deep is the output of an earlier deepening, not a starting point
EXCLUDE = {"sa_rl_deep"}


class Perfect(Exception):
    """A zero-timing-penalty allocation turned up. Carries it out of the search."""

    def __init__(self, alloc):
        super().__init__("zero timing penalty")
        self.alloc = dict(alloc)


class Tracker:
    """The model, with the search objective relabelled and a recorder attached.

    Delegates everything else, so `anneal` and `seed_for` see the real model.
    `Model.evaluate` builds a fresh result dict on every call, so rewriting one
    field of it affects nothing outside this search.

    The recorder lives here rather than in `anneal` so that it sees every
    evaluation, which honours the budget ladder however the rounds fall, and so
    that it survives a round abandoned part way through when a perfect
    allocation turns up.

    Best-so-far tracked here is the same allocation `anneal` would keep: a
    candidate cheaper than the current best is always accepted by the
    Metropolis rule, so the minimum over evaluated candidates and over accepted
    ones are the same thing.
    """

    def __init__(self, model, pref_weight=PREF_WEIGHT, ladder=LADDER,
                 evals=0, curve=None):
        self._model = model
        self._w = pref_weight
        self.evals = evals
        self.curve = dict(curve or {})
        self._marks = [m for m in sorted(ladder) if m > evals]
        self.best_obj = float("inf")
        self.best_hard = float("inf")
        self.best_alloc = None
        self.armed = False

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _offer(self, alloc, hard, total, obj):
        if obj < self.best_obj:
            self.best_obj, self.best_hard = obj, hard
            self.best_total = total
            self.best_alloc = dict(alloc)

    def prime(self, alloc):
        """Score a starting allocation without charging it or arming the stop."""
        res = self._model.evaluate(alloc)
        p = res["penalty"]
        h = p["misses"] + p["budgets"]
        self._offer(alloc, h, h + p["preference"], h + self._w * p["preference"])
        return h

    def snapshot(self, upto=None):
        """Fill ladder points, up to `upto` or to the current evaluation count."""
        limit = self.evals if upto is None else upto
        while self._marks and self._marks[0] <= limit:
            m = self._marks.pop(0)
            self.curve[str(BASE_BUDGET + m)] = {
                "hard": round(self.best_hard, 4),
                "total": round(self.best_total, 4),
                "perfect": self.best_hard <= 1e-9,
            }

    def evaluate(self, alloc):
        res = self._model.evaluate(alloc)
        p = res["penalty"]
        h = p["misses"] + p["budgets"]
        total = h + p["preference"]
        obj = h + self._w * p["preference"]
        p["total"] = obj                      # what the search reads
        self.evals += 1
        self._offer(alloc, h, total, obj)
        self.snapshot()
        if self.armed and h <= 1e-9:
            raise Perfect(alloc)
        return res


def read_start(name):
    """The saved allocations and penalties for one allocator, eval split only."""
    f = ALLOC / f"{name}.jsonl"
    out = {}
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("set") not in (None, splits.EVAL):
                continue
            s = r["summary"]
            out[r["taskset_id"]] = {
                "platform": r["platform_id"],
                "allocation": r["allocation"],
                "hard": s["penalty_hard"],
                "total": s["penalty_total"],
                "n_tasks": s["n_tasks"],
            }
    return out


def last_per_pair(path):
    """Most recent row per (allocator, task set), so a later row supersedes."""
    out = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a line cut in half by ^C
            out[(r["allocator"], r["taskset"])] = r
    return out


def deepen(tracker, alloc, cap, round_evals, round0, on_round):
    """Anneal in reheated rounds until perfect or the step budget is spent.

    `tracker` carries the evaluation count and the curve, including what a
    previous run on this pair spent, so the budget and the ladder are
    cumulative across restarts. `round0` continues the round numbering, so a
    resumed search draws fresh random streams rather than repeating rounds.
    """
    tracker.prime(alloc)
    cur, done, t0 = dict(alloc), round0, time.perf_counter()
    solved = tracker.best_hard <= 1e-9

    while not solved and tracker.evals < cap:
        iters = min(round_evals, cap - tracker.evals)
        tracker.armed = True
        try:
            anneal(tracker, cur, iters=iters, stall=None, checkpoints=None,
                   rng=random.Random(seed_for(tracker) + done))
        except Perfect:
            solved = True
        finally:
            tracker.armed = False
        cur = tracker.best_alloc
        done += 1
        on_round(done, tracker.evals, tracker.best_hard, tracker.best_total,
                 cur, tracker.curve)

    # a search that stopped early holds its result at every larger budget,
    # since best-so-far cannot get worse
    tracker.snapshot(upto=max(LADDER))
    return {
        "allocation": cur,
        "hard": round(tracker.best_hard, 4),
        "total": round(tracker.best_total, 4),
        "solved": solved,
        "rounds_total": done,
        "evals_total": tracker.evals,
        "seconds": round(time.perf_counter() - t0, 3),
        "curve": tracker.curve,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, metavar="ALLOCATOR",
                    help="just these starting points, e.g. sa_bfd sa_wfd. "
                         "Default is every sa_*.jsonl in data/allocations")
    ap.add_argument("--evals", type=int, default=EVALS,
                    help="extra evaluations per instance, past the 4,000 the "
                         "thesis run already spent")
    ap.add_argument("--round-evals", type=int, default=ROUND_EVALS,
                    help="evaluations per reheated round")
    ap.add_argument("--pref-weight", type=float, default=PREF_WEIGHT,
                    help="weight on the core-class preference term in the "
                         "search objective. 0 drops it, 1 is the thesis objective")
    ap.add_argument("--n", type=int, default=None,
                    help="only this many instances per allocator, spread over "
                         "the pool")
    ap.add_argument("--tasksets", nargs="+", default=None, metavar="TASKSET",
                    help="just these task sets, by id. Overrides --n")
    ap.add_argument("--extend", action="store_true",
                    help="reopen pairs already recorded as unsolved and give "
                         "them another --evals from where they stopped")
    ap.add_argument("--to", type=int, default=None, metavar="TOTAL",
                    help="with --extend, carry each pair up to this cumulative "
                         "evaluation count instead of adding --evals to what it "
                         "already spent. Use it to say 'take everything to "
                         "500,000' without working out the difference")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.to and not args.extend:
        raise SystemExit("--to only means something with --extend. For a first "
                         "pass use --evals.")

    names = args.only or sorted(f.stem for f in ALLOC.glob("sa_*.jsonl")
                                if f.stem not in EXCLUDE)
    missing = [n for n in names if not (ALLOC / f"{n}.jsonl").exists()]
    if missing:
        raise SystemExit("no allocation file for: " + ", ".join(missing))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(".state.jsonl")
    finished = last_per_pair(out)
    state = last_per_pair(state_path)

    starts, work = {}, []
    for name in names:
        starts[name] = read_start(name)
        pool = [t for t in tasksets.select(keep=splits.is_eval)
                if t in starts[name] and starts[name][t]["hard"] > 1e-9]
        if args.tasksets:
            want = set(args.tasksets)
            pool = [t for t in pool if t in want]
        elif args.n and args.n < len(pool):
            pool = pool[::len(pool) // args.n][:args.n]
        for t in pool:
            row = finished.get((name, t))
            if row is not None and (row["solved"] or not args.extend):
                continue
            # already at or past the absolute target: nothing to add
            if (row is not None and args.to
                    and row.get("evals_total", 0) >= args.to):
                continue
            work.append((name, t))

    biggest = max((s[t]["n_tasks"] for s in starts.values() for t in s),
                  default=0)
    worst_pref = args.pref_weight * MAX_PREF_PER_TASK * biggest
    if worst_pref >= MIN_MISS:
        print(f"WARNING: --pref-weight {args.pref_weight} lets the preference "
              f"term reach {worst_pref:.2f}, at or above the cheapest miss at "
              f"{MIN_MISS}. The search will trade a miss for a better "
              f"core-class fit, which wastes the budget here.\n")

    print(f"{'allocator':<16}{'unsolved':>10}{'to do now':>11}")
    for name in names:
        pool = sum(1 for t in starts[name] if starts[name][t]["hard"] > 1e-9)
        todo = sum(1 for a, _ in work if a == name)
        print(f"{name:<16}{pool:>10,}{todo:>11,}")
    print(f"{'TOTAL':<16}{'':>10}{len(work):>11,}")
    if not work:
        print("\nnothing left. Use --extend to give the unsolved ones more budget.")
        return

    print(f"\nbudget: {args.evals:,} evaluations per instance past the "
          f"{BASE_BUDGET:,} already spent, in rounds of {args.round_evals:,}, "
          f"stopping early when perfect")
    print(f"objective: misses + budget overruns + {args.pref_weight} x preference")
    print(f"recording the best so far at {len(LADDER)} budgets up to "
          f"{BASE_BUDGET + max(LADDER):,}")
    print("\n^C is safe: rerun the same command to carry on.\n")

    exec_us, hw = load_catalogues()
    platforms, models, bar = {}, {}, Progress(len(work), "searches")
    n_solved, n_done, stopped = 0, 0, False

    with out.open("a") as fh, state_path.open("a") as sfh:
        try:
            for name, tid in work:
                s = starts[name][tid]
                pid = s["platform"]
                if pid not in platforms:
                    platforms[pid] = json.loads(
                        (PLATFORMS / f"{pid}.json").read_text())
                if tid not in models:
                    models.clear()      # one task set at a time, keeps memory flat
                    models[tid] = tasksets.load(tid)
                model = Model(models[tid], platforms[pid], exec_us, hw)

                # Where to pick up. A state row past the finished row's round
                # count is a run that died in flight, so it wins.
                prior, resume = finished.get((name, tid)), state.get((name, tid))
                if resume is not None and (prior is None or resume["rounds_done"]
                                           > prior.get("rounds_total", 0)):
                    alloc, spent = resume["allocation"], resume["evals"]
                    round0, curve = resume["rounds_done"], resume["curve"]
                    cap = resume.get("cap") or args.evals
                    if args.extend:
                        cap = args.to or (cap + args.evals)
                elif prior is not None and args.extend:
                    alloc, spent = prior["allocation"], prior.get("evals_total", 0)
                    round0, curve = prior.get("rounds_total", 0), prior["curve"]
                    cap = args.to or (spent + args.evals)
                else:
                    alloc, spent, round0, curve = s["allocation"], 0, 0, {}
                    cap = args.evals

                tracker = Tracker(model, args.pref_weight, LADDER, spent, curve)

                def checkpoint(rounds_done, evals, h, total, a, cur_curve,
                               _n=name, _t=tid, _cap=cap):
                    sfh.write(json.dumps({
                        "allocator": _n, "taskset": _t,
                        "rounds_done": rounds_done, "evals": evals, "cap": _cap,
                        "hard": round(h, 4), "total": round(total, 4),
                        "curve": cur_curve, "allocation": a}) + "\n")
                    sfh.flush()

                r = deepen(tracker, alloc, cap, args.round_evals, round0,
                           checkpoint)
                g = models[tid]["generator"]
                fh.write(json.dumps({
                    "allocator": name, "taskset": tid, "platform": pid,
                    "n_tasks": len(models[tid]["tasks"]),
                    "n_cores": len(model.core),
                    "load": g["load_level"], "coupling": g["coupling_level"],
                    "start_hard": s["hard"], "start_total": s["total"],
                    **r}) + "\n")
                fh.flush()
                n_solved += r["solved"]
                n_done += 1
                bar.tick(n_done, f"{n_solved:,} solved")
        except KeyboardInterrupt:
            stopped = True

    bar.close()
    if stopped:
        print("\nstopped by ^C. Rerun the same command to carry on: the search "
              "in flight resumes from its last completed round.")
    print(f"\n{n_solved:,} of the {n_done:,} searches finished this run reached "
          f"zero timing penalty")
    print(f"wrote {out}")
    print(f"state {state_path}")


if __name__ == "__main__":
    main()
