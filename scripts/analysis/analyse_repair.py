#!/usr/bin/env python3
"""Analyse the core failure experiment.

    python scripts/analysis/analyse_repair.py
    python scripts/analysis/analyse_repair.py --by load
    python scripts/analysis/analyse_repair.py --csv out/repair_summary.csv

Reads out/runs/repair.jsonl, one row per (task set, dead core, allocator, how many
survivors were freed). Streams it, because the file is a few hundred megabytes
and nothing here needs two rows at once.

The axis this experiment exists to measure is not penalty. It is **migrations**:
how many running tasks had to stop and restart somewhere else. The objective
charges nothing for that and a live system charges a great deal, so every table
carries both and the last section divides one by the other.

Sections:

  0  what was run, and what the system was scoring before the core died
  1  the spectrum, per allocator: from orphans only to a full rebuild
  2  the two ends side by side, so the allocators can be compared
  3  what ten evaluations of search bought on top of each reaction
  4  whether freeing survivors helps at all, measured rather than assumed
  5  what each extra migration is worth
  6  the rebuild against its own search budget, on the same budgets the
     offline search experiment uses, which is where the crossover lives
  7  the two ends broken down by difficulty
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out" / "runs"

LADDER = ["0", "1", "2", "4", "8", "16", "32", "rebuild"]


def mean(v):
    return statistics.mean(v) if v else 0.0


def median(v):
    return statistics.median(v) if v else 0.0


class Acc:
    """Running sums. The file is too big to keep and none of this needs to be."""

    def __init__(self):
        self.n = 0
        self.sums = defaultdict(float)
        self.ms = []
        self.sms = []
        # only the rebuild rows carry one: what the search had at each budget
        self.curve = defaultdict(lambda: defaultdict(float))
        self.curve_n = defaultdict(int)

    def add(self, r):
        self.n += 1
        for b, v in (r.get("sa_curve") or {}).items():
            self.curve_n[b] += 1
            for k in ("hard", "ms", "moved"):
                if k in v:
                    self.curve[b][k] += v[k]
            self.curve[b]["perfect"] += float(v["perfect"])
        self.sums["pen"] += r["penalty_after"]
        self.sums["mig"] += r["migrations"]
        self.sums["spen"] += r["searched_penalty"]
        self.sums["smig"] += r["searched_migrations"]
        self.sums["before"] += r["penalty_before"]
        self.sums["orphans"] += r["orphans"]
        self.sums["tasks"] += r["n_tasks"]
        self.sums["perfect"] += float(r["penalty_after"] <= 1e-9)
        self.sums["sperfect"] += float(r["searched_penalty"] <= 1e-9)
        # kept in full only for the percentiles, which a running sum cannot give
        self.ms.append(r["ms"])
        self.sms.append(r["searched_ms"])

    def avg(self, key):
        return self.sums[key] / self.n if self.n else 0.0

    def pct(self, key):
        return 100 * self.avg(key)


def load(path, by=None):
    """Aggregate the rows. `by` adds a second key, load level or coupling."""
    whole, split = defaultdict(Acc), defaultdict(Acc)
    n = 0
    with Path(path).open() as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["allocator"], r["freed"])
            whole[key].add(r)
            if by:
                split[(key, r[by])].add(r)
            n += 1
    return whole, split, n


def allocators_in(whole):
    seen, out = set(), []
    for a, _ in whole:
        if a not in seen and a != "anneal":
            seen.add(a)
            out.append(a)
    # the policy first, it is the subject
    out.sort(key=lambda a: (a != "rl", a))
    return out + (["anneal"] if any(a == "anneal" for a, _ in whole) else [])


def ladder_of(whole, a):
    return [k for k in LADDER if (a, k) in whole]


def section0(whole, n):
    any_acc = next(iter(whole.values()))
    fails = sum(v.n for k, v in whole.items() if k == next(iter(whole)))
    print(f"{'=' * 100}\ncore failure experiment\n{'=' * 100}")
    print(f"{n:,} rows, {any_acc.n:,} failures, "
          f"{any_acc.avg('orphans'):.1f} orphans of {any_acc.avg('tasks'):.0f} "
          f"tasks on average")
    print(f"penalty before the failure: {any_acc.avg('before'):.2f}   "
          f"(the allocation that was running)")


def section1(whole, allocators):
    print(f"\n{'=' * 100}\nthe spectrum, per allocator\n{'=' * 100}")
    print("K is how many surviving tasks the allocator was allowed to move, on "
          "top of the\norphans the dead core stranded. 0 is pure repair, "
          "rebuild is everything.\n")
    for a in allocators:
        ks = ladder_of(whole, a)
        if len(ks) < 2:
            continue
        print(f"  {a}")
        print(f"  {'K':<10}{'penalty':>10}{'perfect':>9}{'moved':>8}"
              f"{'% tasks':>9}{'med ms':>9}{'p90 ms':>9}{'vs K=0':>10}")
        # signed against the pure repair end rather than scaled between the two
        # ends, since scaling would assume a rebuild is the better end
        lo = whole[(a, ks[0])].avg("pen")
        for k in ks:
            v = whole[(a, k)]
            ms = sorted(v.ms)
            d = v.avg("pen") - lo
            print(f"  {k:<10}{v.avg('pen'):>10.1f}{v.pct('perfect'):>8.1f}%"
                  f"{v.avg('mig'):>8.1f}{100 * v.avg('mig') / v.avg('tasks'):>8.0f}%"
                  f"{median(v.ms):>9.2f}{ms[int(0.9 * (len(ms) - 1))]:>9.2f}"
                  f"{d:>+10.1f}")
        print()


def section2(whole, allocators):
    print(f"{'=' * 100}\nthe two ends, side by side\n{'=' * 100}")
    for end, what in (("0", "repair: only the stranded tasks move"),
                      ("rebuild", "rebuild: every task is reallocated")):
        rows = [(a, whole[(a, end)]) for a in allocators if (a, end) in whole]
        if not rows:
            continue
        print(f"\n  {what}")
        print(f"  {'allocator':<12}{'penalty':>10}{'perfect':>9}{'moved':>8}"
              f"{'% tasks':>9}{'med ms':>9}")
        for a, v in sorted(rows, key=lambda x: x[1].avg("pen")):
            print(f"  {a:<12}{v.avg('pen'):>10.1f}{v.pct('perfect'):>8.1f}%"
                  f"{v.avg('mig'):>8.1f}"
                  f"{100 * v.avg('mig') / v.avg('tasks'):>8.0f}%"
                  f"{median(v.ms):>9.2f}")


def section3(whole, allocators):
    print(f"\n{'=' * 100}\nwhat ten evaluations of search bought\n{'=' * 100}")
    print("the same reaction, then a short anneal. The budget is a failure "
          "window, not a\nsearch budget, and it is identical for every row.\n")
    print(f"  {'reaction':<18}{'penalty':>10}{'-> searched':>13}{'moved':>8}"
          f"{'-> searched':>13}{'med ms':>9}{'-> searched':>13}")
    for a in allocators:
        for k in ladder_of(whole, a):
            if k not in ("0", "rebuild"):
                continue
            v = whole[(a, k)]
            print(f"  {a + ':' + k:<18}{v.avg('pen'):>10.1f}"
                  f"{v.avg('spen'):>13.1f}{v.avg('mig'):>8.1f}"
                  f"{v.avg('smig'):>13.1f}{median(v.ms):>9.2f}"
                  f"{median(v.sms):>13.2f}")


def section4(whole, allocators):
    """Does being allowed to move more running tasks help at all?

    The question the experiment was built to answer assumed it would, and that
    somewhere on the way there would be a knee worth stopping at. Whether that
    is true is a property of the data, so it is measured here rather than
    assumed: find the K with the lowest penalty and say where it landed.
    """
    print(f"\n{'=' * 100}\ndoes freeing survivors help\n{'=' * 100}")
    print("the best K for each allocator, and what a full rebuild costs "
          "relative to pure\nrepair. A positive rebuild column means "
          "reallocating everything made it worse.\n")
    print(f"  {'allocator':<12}{'best K':>8}{'its penalty':>13}"
          f"{'vs K=0':>10}{'extra moves':>13}{'rebuild vs K=0':>17}")
    for a in allocators:
        ks = ladder_of(whole, a)
        if len(ks) < 2:
            continue
        lo = whole[(a, ks[0])]
        best = min(ks, key=lambda k: whole[(a, k)].avg("pen"))
        v = whole[(a, best)]
        rb = whole[(a, "rebuild")].avg("pen") - lo.avg("pen") \
            if (a, "rebuild") in whole else float("nan")
        print(f"  {a:<12}{best:>8}{v.avg('pen'):>13.1f}"
              f"{v.avg('pen') - lo.avg('pen'):>+10.1f}"
              f"{v.avg('mig') - lo.avg('mig'):>+13.1f}"
              f"{(f'{rb:+.1f}' if rb == rb else '-'):>17}")
    print("\n  best K = 0 means nothing was gained by disturbing a single "
          "running task\n  beyond the ones the dead core stranded.")


def section5(whole, allocators):
    print(f"\n{'=' * 100}\nwhat each extra migration is worth\n{'=' * 100}")
    print("penalty change per extra task moved, against pure repair. Negative "
          "is an\nimprovement, positive means the migration made things worse. "
          "Nothing in the\nobjective charges for a migration, so this is the "
          "number a live system needs.\n")
    print(f"  {'reaction':<18}{'penalty':>10}{'vs K=0':>10}{'extra moves':>13}"
          f"{'per move':>11}")
    for a in allocators:
        ks = ladder_of(whole, a)
        if not ks:
            continue
        base = whole[(a, ks[0])]
        for k in ks:
            v = whole[(a, k)]
            d = v.avg("pen") - base.avg("pen")
            extra = v.avg("mig") - base.avg("mig")
            per = d / extra if extra > 1e-9 else float("nan")
            print(f"  {a + ':' + k:<18}{v.avg('pen'):>10.1f}{d:>+10.1f}"
                  f"{extra:>+13.1f}"
                  f"{(f'{per:+.1f}' if per == per else '-'):>11}")
        print()


def section_rebuild_curve(whole, allocators):
    """What a rebuild had after B evaluations, on the offline experiment's budgets.

    A rebuild is the offline pipeline run again on a platform with one core
    fewer, so it is the one rung of the ladder that can be read against the
    search budget results directly. It is also the rung that answers the
    question a live system actually has: how long do I have to give this before
    it beats simply patching the hole.
    """
    have = [a for a in allocators if (a, "rebuild") in whole
            and whole[(a, "rebuild")].curve_n]
    if not have:
        return
    marks = sorted({int(b) for a in have
                    for b in whole[(a, "rebuild")].curve_n}, key=int)

    print(f"\n{'=' * 100}\nrebuild against its search budget\n{'=' * 100}")
    print("B = 0 is the one pass allocation the search started from. Pure "
          "repair is on the\nfar right for comparison, and it costs "
          "milliseconds rather than a second.\n")
    for field, lab, fmt in (("hard", "mean penalty", "{:>10.1f}"),
                            ("perfect", "perfect rate", "{:>9.0f}%"),
                            ("moved", "tasks moved", "{:>10.1f}"),
                            ("ms", "mean ms", "{:>10.1f}")):
        print(f"  {lab}")
        print(f"  {'allocator':<12}" + "".join(f"{b:>10}" for b in marks)
              + f"{'| repair':>12}")
        for a in have:
            acc = whole[(a, "rebuild")]
            line = f"  {a:<12}"
            for b in marks:
                n = acc.curve_n[str(b)]
                if not n:
                    line += f"{'-':>10}"
                    continue
                v = acc.curve[str(b)][field] / n
                line += (f"{100 * v:>9.0f}%" if field == "perfect"
                         else fmt.format(v))
            rep = whole.get((a, "0"))
            if rep is None:
                line += f"{'-':>12}"
            elif field == "perfect":
                line += f"{rep.pct('perfect'):>11.0f}%"
            elif field == "moved":
                line += f"{rep.avg('mig'):>12.1f}"
            elif field == "ms":
                line += f"{mean(rep.ms):>12.2f}"
            else:
                line += f"{rep.avg('pen'):>12.1f}"
            print(line)
        print()


def section_by(split, allocators, axis):
    print(f"\n{'=' * 100}\nby {axis}\n{'=' * 100}")
    values = sorted({v for (_, v) in split})
    for end in ("0", "rebuild"):
        print(f"\n  mean penalty, K = {end}")
        print(f"  {'allocator':<12}" + "".join(f"{str(v):>11}" for v in values))
        for a in allocators:
            line = f"  {a:<12}"
            for v in values:
                acc = split.get(((a, end), v))
                line += f"{acc.avg('pen'):>11.1f}" if acc else f"{'-':>11}"
            print(line)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=str(OUT / "repair.jsonl"))
    ap.add_argument("--by", choices=["load", "coupling"], default="load",
                    help="second axis for the breakdown section")
    ap.add_argument("--csv", help="write the per reaction summary as csv")
    args = ap.parse_args()

    if not Path(args.rows).exists():
        raise SystemExit(f"no {args.rows}. run scripts/run/repair.py first")

    whole, split, n = load(args.rows, args.by)
    allocators = allocators_in(whole)

    section0(whole, n)
    section1(whole, allocators)
    section2(whole, allocators)
    section3(whole, allocators)
    section4(whole, allocators)
    section5(whole, allocators)
    section_rebuild_curve(whole, allocators)
    section_by(split, allocators, args.by)

    if args.csv:
        import csv
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["allocator", "freed", "n", "penalty", "perfect_pct",
                        "migrations", "median_ms", "searched_penalty",
                        "searched_migrations", "searched_median_ms"])
            for (a, k), v in whole.items():
                w.writerow([a, k, v.n, round(v.avg("pen"), 4),
                            round(v.pct("perfect"), 2), round(v.avg("mig"), 3),
                            round(median(v.ms), 3), round(v.avg("spen"), 4),
                            round(v.avg("smig"), 3), round(median(v.sms), 3)])
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
