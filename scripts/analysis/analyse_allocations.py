"""Analyse the saved allocations.

    python scripts/analysis/analyse_allocations.py --eval --common   the headline numbers
    python scripts/analysis/analyse_allocations.py --pair bfd rl
    python scripts/analysis/analyse_allocations.py --cells > out/cells.txt   the long form
    python scripts/analysis/analyse_allocations.py --csv out/results.csv

Reads data/allocations/<allocator>.jsonl. Any number of allocators, whichever
files happen to be there.

Three allocators in the tables are not methods. `best_greedy` and
`best_greedy_sa` are the best of the six heuristics on each instance, which is
the baseline the comparison has to beat. Their time is the sum of all six,
since all six had to run. `oracle` is the best anything managed, per instance.

Sections:

  0  pools. Which allocators are present and over how many task sets. The
     annealed ones are usually run on a subset, so the pools rarely match.
  1  per allocator, per (platform, coupling, load): perfect rate and penalty
  2  the marginals: perfect rate and penalty against difficulty, dependency,
     devices, cores, link speed, and which held out question the instance
     answers. Allocator down the side, setting across.
  c  chain budgets. Overruns split out of penalty_hard, with the headroom
     that caused them, since every other table folds them into one number.
  t  allocation time, per allocator and per platform, plus how the exact
     allocator spent it. A leaderboard without this puts a packer and a search
     in the same column with no hint that one is a thousand times slower.
  b  the search budget curve, read off the annealed runs themselves.
  3  head to head. A leaderboard, one pair broken down by axis, and what
     annealing bought over its own starting point. The full all pairs matrix is
     eighteen columns wide and only prints with --matrix.
  v  every allocator against one reference, with a bootstrap interval and a
     sign test. This is the column of the matrix anyone actually reads.
  o  how far from optimal, measured against the exact allocator wherever it
     settled the instance. The only section that can say how good is good.
  4  the flatness report: which cells are trivial, which are hopeless, and which
     actually separate the allocators. This is the one that drives tuning.

Penalty is `internal_weight x external_weight` per missed TASK, charged once per
task and not once per job, so a 100 us task that misses is worth its weight once
over the hyperperiod rather than twenty times.

`penalty_hard` is misses plus budget, without the preference term. Wins are
counted on that, because preference is a tiebreaker and would otherwise decide
instances where nothing failed at all.

Every paired comparison intersects on task set id first, so allocators run over
different pools are still compared only on what both of them saw.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import splits
from allocators import QUALITY_JUDGES, base_of

ROOT = Path(__file__).resolve().parent.parent.parent
ALLOC = ROOT / "data" / "allocations"
PLATFORMS = ROOT / "data" / "platforms"
SPLITS = splits.SPLITS

# Allocators that are not a single method. `best_*` is what an engineer with no
# policy would do: run every heuristic and keep whichever came out best, so it
# is the baseline the comparison has to beat. `oracle` is the best anything
# found per instance, the ceiling of the table.
GREEDY = ("bfd", "wfd", "bfd_crit", "wfd_crit", "heft", "heft_crit")
PORTFOLIOS = {
    "best_greedy": GREEDY,
    "best_greedy_sa": tuple(f"sa_{g}" for g in GREEDY),
}
ORACLE = "oracle"
VIRTUAL = tuple(PORTFOLIOS) + (ORACLE,)

# A cell is wasted when the allocators cannot be told apart on it, not when they
# fail to be perfect. Imperfect instances are the point: the objective is to
# minimise penalty, so a cell where everyone misses but by different amounts is
# more informative than one where everyone is perfect.
TIED = "tied"              # allocators produce the same penalty on most instances
OVER = "oversubscribed"    # mean effective core load above 1.0, no allocation fits
USEFUL = "useful"

TIE_RATE_WASTED = 0.80     # above this a cell tells you nothing about the allocators
OVERSUBSCRIBED = 1.5       # mean effective load: demand this far past capacity cannot fit

# the all pairs matrix is 18 columns wide, so it is off unless asked for and
# `section_vs` prints the one column against a reference instead
MATRIX = False


def platform_meta():
    """Shape of every platform, so results can be grouped by hardware.

    Device count, core count, CPU type count and the slowest link are what a
    difference can be attributed to; the platform id alone is not.
    """
    meta = {}
    for f in sorted(PLATFORMS.glob("platform_*.json")):
        p = json.loads(f.read_text())
        links = [lk["link_type"] for lk in p["links"]]
        rank = {"eth_10g": 0, "eth_1g": 1, "usb": 2}
        meta[p["platform_id"]] = {
            "devices": len(p["devices"]),
            "cores": len(p["cores"]),
            "cpu_types": len({c["cpu_type"] for c in p["cores"]}),
            # the worst link bounds how expensive a crossing can get, so it is
            # the one that characterises the wiring
            "link": max(links, key=lambda l: rank.get(l, 9)) if links else "none",
        }
    return meta


def pool_of():
    """Which held out question each task set answers.

    `workload` is an unseen task set on a platform the policy trained on,
    `hardware` one of the five platforms it never saw. One mean over both
    answers neither question, so every table can separate them.
    """
    out = {}
    for name in ("test_workload", "test_hardware"):
        f = SPLITS / f"{name}.txt"
        if f.exists():
            for t in f.read_text().split():
                out[t] = name.replace("test_", "")
    return out


def load():
    meta, pools = platform_meta(), pool_of()
    rows = []
    for f in sorted(ALLOC.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                pid = r["platform_id"]
                rows.append({"allocator": r["allocator"], "taskset": r["taskset_id"],
                             # written by run_allocators. Older files predate the
                             # key, so fall back to the split itself
                             "set": r.get("set") or splits.set_of(r["taskset_id"]),
                             "platform": pid,
                             "pool": pools.get(r["taskset_id"], "train"),
                             "coupling": r["coupling_level"], "load": r["load_level"],
                             **meta.get(pid, {}),
                             **r["summary"]})
    return rows


def add_portfolios(rows):
    """Synthesise the portfolio and oracle allocators, one row per task set.

    A portfolio is the best of several that were all run, so its row is
    whichever member won on that instance and its time is the sum of all of
    them rather than the winner's alone.
    """
    by = defaultdict(dict)
    for r in rows:
        by[r["taskset"]][r["allocator"]] = r

    made = []
    for tid, got in by.items():
        for name, members in PORTFOLIOS.items():
            have = [got[m] for m in members if m in got]
            if len(have) < 2:
                continue
            made.append(_pick(have, name))
        real = [r for r in got.values() if r["allocator"] not in VIRTUAL]
        if len(real) > 1:
            made.append(_pick(real, ORACLE))
    return rows + made


def _pick(members, name):
    best = min(members, key=lambda r: (r["penalty_hard"], r["penalty_total"]))
    ms = [m.get("alloc_ms") for m in members]
    # a portfolio has no search and no proof of its own, so everything method
    # specific is dropped and only the result survives
    out = {k: v for k, v in best.items()
           if not k.startswith("sa_") and not k.startswith("exact_")}
    out["allocator"] = name
    out["alloc_ms"] = (sum(m for m in ms if m is not None)
                       if any(m is not None for m in ms) else None)
    return out


def mean(v):
    return sum(v) / len(v) if v else 0.0


def median(v):
    s = sorted(v)
    return s[len(s) // 2] if s else 0.0


def pnum(p):
    return int(p.split("_")[1])


def order_allocators(present):
    """Base allocators first, then each one's annealed twin, where alphabetical
    would put sa_bfd between bfd and wfd. The portfolios and the oracle go
    last, being references rather than methods."""
    bases = [a for a in sorted(present)
             if not a.startswith("sa_") and a not in VIRTUAL]
    out = []
    for b in bases:
        out.append(b)
        if f"sa_{b}" in present:
            out.append(f"sa_{b}")
    out += [a for a in sorted(present) if a not in out and a not in VIRTUAL]
    return out + [a for a in VIRTUAL if a in present]


def section0(rows, allocators):
    print(f"{'=' * 104}\npools\n{'=' * 104}")
    print(f"{'allocator':<16}{'task sets':>11}{'train':>8}{'eval':>8}"
          f"{'searched':>10}{'mean evals':>12}{'accept':>9}")
    counts = {}
    for a in allocators:
        cell = [r for r in rows if r["allocator"] == a]
        counts[a] = len(cell)
        ev = [r["sa_evals"] for r in cell if "sa_evals" in r]
        acc = [r["sa_accept_rate"] for r in cell if "sa_accept_rate" in r]
        searched = "yes" if ev else "no"
        n_ev = f"{mean(ev):.0f}" if ev else "-"
        rate = f"{mean(acc):.2f}" if acc else "-"
        print(f"{a:<16}{len(cell):>11}"
              f"{sum(1 for r in cell if r['set'] == splits.TRAIN):>8}"
              f"{sum(1 for r in cell if r['set'] == splits.EVAL):>8}"
              f"{searched:>10}{n_ev:>12}{rate:>9}")

    if len(set(counts.values())) > 1:
        common = None
        by_ts = defaultdict(set)
        for r in rows:
            by_ts[r["taskset"]].add(r["allocator"])
        common = sum(1 for v in by_ts.values() if len(v) == len(allocators))
        print(f"\n  pools differ. {common} task sets were seen by every allocator.")
        print("  paired sections already intersect. --common applies it everywhere.")


def section1(rows, allocators, platforms):
    for a in allocators:
        print(f"\n{'=' * 104}\n{a}   per platform, coupling, load\n{'=' * 104}")
        print(f"{'platform':<11}{'cpl':>4}{'load':>5}{'n':>5}{'perfect':>10}"
              f"{'penalty mean':>14}{'median':>9}{'p90':>9}"
              f"{'H/F/S misses':>15}{'cross':>7}")
        for p in platforms:
            for c in sorted({r["coupling"] for r in rows}):
                for l in sorted({r["load"] for r in rows}):
                    cell = [r for r in rows if r["allocator"] == a
                            and r["platform"] == p and r["coupling"] == c
                            and r["load"] == l]
                    if not cell:
                        continue
                    pen = sorted(r["penalty_hard"] for r in cell)
                    nperf = sum(r["perfect"] for r in cell)
                    print(f"{p:<11}{c:>4}{l:>5}{len(cell):>5}"
                          f"{f'{100 * nperf / len(cell):.0f}%':>10}"
                          f"{mean(pen):>14.1f}{median(pen):>9.1f}"
                          f"{pen[int(0.9 * (len(pen) - 1))]:>9.1f}"
                          f"{sum(r['missed_hard'] for r in cell):>7}"
                          f"/{sum(r['missed_firm'] for r in cell)}"
                          f"/{sum(r['missed_soft'] for r in cell):<5}"
                          f"{mean([r['n_crossings'] for r in cell]):>7.1f}")


AXES = [("load", "difficulty (load level)", None),
        ("coupling", "dependency (coupling level)", None),
        ("devices", "devices on the platform", None),
        ("cores", "cores on the platform", None),
        ("link", "slowest link on the platform", None),
        ("pool", "held out question", None),
        ("platform", "platform", pnum)]

METRICS = [("perfect", "perfect rate (%)",
            lambda c: 100 * mean([float(r["perfect"]) for r in c]), "{:>8.0f}"),
           ("penalty", "mean penalty_hard",
            lambda c: mean([r["penalty_hard"] for r in c]), "{:>8.1f}")]


def grouped(rows, allocators, axis, key, metric, width=9):
    """One table: allocator down the side, axis values across.

    With eighteen allocators, putting them across the top makes every table
    wider than a terminal. Down the side they fit, and comparing two of them at
    the same setting reads down a column.
    """
    values = sorted({r[axis] for r in rows if axis in r},
                    key=key or (lambda x: (isinstance(x, str), x)))
    if len(values) < 2:
        return
    print(f"  {'allocator':<16}" + "".join(f"{str(v):>{width}}" for v in values))
    for a in allocators:
        line = f"  {a:<16}"
        for v in values:
            cell = [r for r in rows if r["allocator"] == a and r.get(axis) == v]
            line += (metric[3].format(metric[2](cell)) if cell
                     else f"{'-':>{width}}")
        print(line)
    print()


def section2(rows, allocators, axes=None):
    print(f"\n{'=' * 104}\nmarginals: which axis actually moves the result\n{'=' * 104}")
    want = set(axes) if axes else {a for a, _, _ in AXES}
    for axis, label, key in AXES:
        if axis not in want or axis not in rows[0] and axis != "platform":
            continue
        for metric in METRICS:
            print(f"  {label}  ->  {metric[1]}")
            grouped(rows, allocators, axis, key, metric,
                    width=11 if axis in ("link", "pool") else 9)


def bootstrap_ci(diffs, reps=2000, level=0.95, seed=0):
    """Percentile bootstrap on the mean paired difference.

    Penalty is heavy tailed, most instances costing nothing and a few costing
    hundreds, so a normal interval would be wrong. Resampling the paired
    differences assumes no distribution, and pairing removes the instance to
    instance variance.
    """
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((1 - level) / 2 * reps)]
    hi = means[min(reps - 1, int((1 + level) / 2 * reps))]
    return lo, hi


def sign_test(w, l):
    """Two sided probability of a win/loss split this lopsided by chance alone.

    Ties are dropped, which is the standard sign test. Normal approximation,
    fine at these counts. Returned as a p value.
    """
    n = w + l
    if n == 0:
        return 1.0
    z = abs(w - l) / math.sqrt(n)
    return math.erfc(z / math.sqrt(2))


def paired(by, a, b, sel=lambda r: True, key="penalty_hard"):
    """wins, ties, losses, mean difference. Negative difference favours a."""
    w = t = l = 0
    diffs = []
    for pair in by.values():
        if a not in pair or b not in pair or not sel(pair[a]):
            continue
        pa, pb = pair[a][key], pair[b][key]
        diffs.append(pa - pb)
        if abs(pa - pb) < 1e-9:
            t += 1
        elif pa < pb:
            w += 1
        else:
            l += 1
    return w, t, l, diffs


def section_chains(rows, allocators):
    """What the chain budget is costing.

    `penalty_hard` folds misses and budget overruns into one number, so a
    budget problem is invisible in every other table. Here they are separated,
    with the latency over budget that produced them, where 1.0 is exactly on
    the limit.
    """
    print(f"\n{'=' * 104}\nchain budgets\n{'=' * 104}")
    print("latency / budget is the whole chain, so 1.0 is exactly on the limit "
          "and above it is a violation.\n")
    print(f"{'allocator':<16}{'n':>7}{'violated':>10}{'pen budget':>12}"
          f"{'pen miss':>10}{'budget share':>14}{'lat/budget p50':>16}{'p90':>8}")
    for a in allocators:
        cell = [r for r in rows if r["allocator"] == a]
        if not cell:
            continue
        ratios = sorted(r["chain_latency_us"] / r["chain_budget_us"] for r in cell
                        if r.get("chain_budget_us") and r.get("chain_latency_us"))
        b = mean([r["penalty_budgets"] for r in cell])
        m = mean([r["penalty_misses"] for r in cell])
        share = 100 * b / (b + m) if (b + m) else 0.0
        p50 = ratios[len(ratios) // 2] if ratios else 0.0
        p90 = ratios[min(len(ratios) - 1, int(0.9 * len(ratios)))] if ratios else 0.0
        viol = 100 * mean([float(bool(r["chain_violated"])) for r in cell])
        print(f"{a:<16}{len(cell):>7}{f'{viol:.1f}%':>10}{b:>12.2f}{m:>10.1f}"
              f"{f'{share:.1f}%':>14}{p50:>16.2f}{p90:>8.2f}")

    print("\n  violated rate by load level")
    loads = sorted({r["load"] for r in rows})
    print("  " + f"{'allocator':<16}" + "".join(f"{f'load {l}':>10}" for l in loads))
    for a in allocators:
        line = f"  {a:<16}"
        for l in loads:
            cell = [r for r in rows if r["allocator"] == a and r["load"] == l]
            line += (f"{100 * mean([float(bool(r['chain_violated'])) for r in cell]):>9.1f}%"
                     if cell else f"{'-':>10}")
        print(line)

    worst = [(mean([r["penalty_budgets"] for r in rows
                    if r["allocator"] == a and r["platform"] == p]), p, a)
             for a in allocators for p in {r["platform"] for r in rows}
             if any(r["allocator"] == a and r["platform"] == p for r in rows)]
    worst = sorted((w for w in worst if w[0] > 0), reverse=True)[:5]
    if worst:
        print("\n  worst (allocator, platform) by mean budget penalty")
        for v, p, a in worst:
            print(f"    {a:<16}{p:<14}{v:>8.2f}")
    else:
        print("\n  no chain budget was violated anywhere")


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


def section_time(rows, allocators, platforms):
    """How long each allocator took.

    Without this a one pass packer and a search sit in the same column of the
    leaderboard with no hint that one is a thousand times slower. Wall clock of
    the allocator call only, recorded by run_allocators; building the Model and
    scoring the result afterwards belong to the harness.

    A row with no `alloc_ms` is left out, so n here can be smaller than in the
    leaderboard rather than the mean being wrong.
    """
    have = {a: [r["alloc_ms"] for r in rows
                if r["allocator"] == a and r.get("alloc_ms") is not None]
            for a in allocators}
    if not any(have.values()):
        print(f"\n{'=' * 104}\nallocation time\n{'=' * 104}")
        print("no timings on file. Rerun scripts/run/run_allocators.py to record them.")
        return

    print(f"\n{'=' * 104}\nallocation time\n{'=' * 104}")
    print("wall clock of the allocator call alone, in milliseconds.\n")
    print(f"{'allocator':<14}{'n':>7}{'mean':>10}{'median':>10}{'p90':>10}"
          f"{'max':>10}{'total s':>11}{'vs bfd':>9}")
    ref = mean(have.get("bfd") or []) or None
    for a in allocators:
        v = have[a]
        if not v:
            print(f"{a:<14}{0:>7}{'-':>10}{'-':>10}{'-':>10}{'-':>10}{'-':>11}{'-':>9}")
            continue
        rel = f"{mean(v) / ref:.0f}x" if ref else "-"
        print(f"{a:<14}{len(v):>7}{mean(v):>10.2f}{median(v):>10.2f}"
              f"{pct(v, 0.9):>10.2f}{max(v):>10.1f}{sum(v) / 1000:>11.1f}{rel:>9}")

    w = max(12, max(len(a) for a in allocators) + 2)
    print("\n  mean ms by platform")
    print(f"  {'platform':<13}" + "".join(f"{a:>{w}}" for a in allocators))
    for p in platforms:
        line = f"  {p:<13}"
        for a in allocators:
            v = [r["alloc_ms"] for r in rows if r["allocator"] == a
                 and r["platform"] == p and r.get("alloc_ms") is not None]
            line += f"{mean(v):>{w}.1f}" if v else f"{'-':>{w}}"
        print(line)

    # the exact allocator's runtime is a result rather than a cost, since what
    # it buys for the time is a proof
    ex = [r for r in rows if r.get("exact_status")]
    if not ex:
        return
    print(f"\n  exact: what the search settled")
    print(f"  {'status':<14}{'n':>7}{'share':>9}{'mean ms':>10}{'mean cost':>11}"
          f"{'mean nodes':>12}")
    for st, what in (
            ("schedulable", "proved: a zero penalty allocation exists"),
            ("infeasible", "proved: no allocation meets every deadline"),
            ("timeout", "unknown: ran out of time, best found only")):
        cell = [r for r in ex if r["exact_status"] == st]
        if not cell:
            continue
        print(f"  {st:<14}{len(cell):>7}{100 * len(cell) / len(ex):>8.1f}%"
              f"{mean([r['exact_ms'] for r in cell]):>10.0f}"
              f"{mean([r['exact_cost'] for r in cell]):>11.2f}"
              f"{mean([r['exact_nodes'] for r in cell]):>12.0f}   {what}")
    proved = sum(1 for r in ex if r["exact_optimal"])
    print(f"\n  settled {proved} of {len(ex)} ({100 * proved / len(ex):.1f}%), "
          f"the rest are unknown rather than unsolvable")
    gained = [r for r in ex if r["exact_improved"] > 1e-9]
    print(f"  beat the best allocation already on file {len(gained)} times"
          + (f", by {mean([r['exact_improved'] for r in gained]):.2f} on average"
             if gained else ""))


def section_budget(rows, allocators):
    """The search budget curve, read off the annealed runs themselves.

    Every annealed allocator writes down its best so far as it passes each
    budget, so one full run answers "what would this have produced with B
    evaluations" for every B, with no separate sweep that could disagree with
    the headline table.
    """
    have = [a for a in allocators
            if any(r.get("sa_curve") for r in rows if r["allocator"] == a)]
    if not have:
        return

    marks = sorted({int(k) for a in have for r in rows
                    if r["allocator"] == a and r.get("sa_curve")
                    for k in r["sa_curve"]})

    print(f"\n{'=' * 104}\nsearch budget\n{'=' * 104}")
    print("what each search had after B evaluations of the timing model. "
          "B = 0 is the allocation\nit started from, before the search touched "
          "it.\n")
    def curve(a, field, sel=lambda r: True):
        cell = [r["sa_curve"] for r in rows
                if r["allocator"] == a and r.get("sa_curve") and sel(r)]
        out = {}
        for b in marks:
            v = [c[str(b)][field] for c in cell if str(b) in c]
            if v:
                out[b] = (100 * mean([float(x) for x in v]) if field == "perfect"
                          else median(v) if field == "ms" else mean(v))
        return out

    for field, label in (("hard", "mean penalty_hard"),
                         ("perfect", "perfect rate"),
                         ("ms", "median ms")):
        print(f"  {label}")
        print(f"  {'allocator':<14}" + "".join(f"{b:>10}" for b in marks))
        for a in have:
            c = curve(a, field)
            line = f"  {a:<14}"
            for b in marks:
                if b not in c:
                    line += f"{'-':>10}"
                elif field == "perfect":
                    line += f"{c[b]:>9.0f}%"
                else:
                    line += f"{c[b]:>10.1f}"
            print(line)
        print()

    # What the starting point is worth, as how much search the others need to
    # reach where it already is. Interpolated in log space between the measured
    # budgets, where the curve is close to straight and the marks an octave
    # apart.
    ref = "sa_rl" if "sa_rl" in have else have[0]
    others = [a for a in have if a != ref]
    if others:
        rc = curve(ref, "hard")
        print(f"  budget each allocator needs to match {ref}")
        print(f"  {'at budget':<14}{'its penalty':>12}"
              + "".join(f"{a:>16}" for a in others))
        for b in marks:
            if b == 0 or b not in rc:
                continue
            line = f"  {b:<14}{rc[b]:>12.1f}"
            for a in others:
                c = curve(a, "hard")
                got = _budget_for(c, marks, rc[b])
                line += (f"{f'{got:.0f} ({got / b:.1f}x)':>16}" if got
                         else f"{'>' + str(marks[-1]):>16}")
            print(line)
        print()

    # Where the advantage comes from. A curve averaged over everything hides
    # that easy instances are solved by any budget and hard ones by none, so
    # the difference lives in the middle.
    loads = sorted({r["load"] for r in rows})
    if len(loads) > 1:
        print(f"  mean penalty_hard by difficulty, {ref} against the field")
        for l in loads:
            print(f"    load {l}")
            for a in have:
                c = curve(a, "hard", lambda r, l=l: r["load"] == l)
                print(f"      {a:<14}"
                      + "".join(f"{c[b]:>10.1f}" if b in c else f"{'-':>10}"
                                for b in marks))
        print()


def _budget_for(c, marks, target):
    """Smallest budget at which this curve is at or below `target`."""
    xs = [(b, c[b]) for b in marks if b in c and b > 0]
    for (b0, v0), (b1, v1) in zip(xs, xs[1:]):
        if v0 >= target >= v1:
            if v0 <= v1 or v0 <= 0 or v1 <= 0 or target <= 0:
                return b0
            f = (math.log(v0) - math.log(target)) / (math.log(v0) - math.log(v1))
            return math.exp(math.log(b0) + f * (math.log(b1) - math.log(b0)))
    return None


def section_optimal(rows, allocators):
    """How far from optimal everything is, where optimal is known.

    The exact allocator settles an instance in one of two ways. `schedulable`
    means it found a zero penalty allocation, so the optimum is zero and any
    penalty an allocator scores there is its whole distance from optimal.
    `infeasible` means it explored the tree to the end and the best possible is
    still above zero, so the optimum is that number. Anything it timed out on
    has no known optimum and is excluded rather than guessed at.
    """
    ex = {r["taskset"]: r for r in rows if r["allocator"] == "exact"
          and r.get("exact_status")}
    if not ex:
        return

    tally = Counter(r["exact_status"] for r in ex.values())
    n = len(ex)
    print(f"\n{'=' * 104}\nhow far from optimal\n{'=' * 104}")
    print(f"the exact allocator settled {tally['schedulable'] + tally['infeasible']}"
          f" of {n} instances ({100 * (tally['schedulable'] + tally['infeasible']) / n:.1f}%)\n")
    for st, what in (("schedulable", "a zero penalty allocation provably exists"),
                     ("infeasible", "provably no allocation meets everything"),
                     ("timeout", "unknown, the search ran out of time")):
        if tally[st]:
            print(f"  {st:<14}{tally[st]:>7}{100 * tally[st] / n:>8.1f}%   {what}")

    known = {t: (0.0 if r["exact_status"] == "schedulable" else r["exact_cost"])
             for t, r in ex.items() if r["exact_status"] != "timeout"}
    if not known:
        print("\n  nothing was settled, so no gap can be measured")
        return

    print(f"\n  against the {len(known)} instances with a known optimum")
    print(f"  {'allocator':<16}{'n':>7}{'optimal':>10}{'mean gap':>11}"
          f"{'median gap':>12}{'p90 gap':>10}{'worst':>10}")
    for a in allocators:
        gaps = [r["penalty_hard"] - known[r["taskset"]] for r in rows
                if r["allocator"] == a and r["taskset"] in known]
        if not gaps:
            continue
        hit = sum(1 for g in gaps if g < 1e-9)
        print(f"  {a:<16}{len(gaps):>7}{100 * hit / len(gaps):>9.1f}%"
              f"{mean(gaps):>11.2f}{median(gaps):>12.2f}"
              f"{pct(gaps, 0.9):>10.2f}{max(gaps):>10.1f}")

    print("\n  proved schedulable, by difficulty")
    loads = sorted({r["load"] for r in ex.values()})
    print(f"  {'':<16}" + "".join(f"{'load ' + str(l):>10}" for l in loads))
    for label, want in (("schedulable", "schedulable"), ("infeasible", "infeasible"),
                        ("unknown", "timeout")):
        line = f"  {label:<16}"
        for l in loads:
            cell = [r for r in ex.values() if r["load"] == l]
            line += (f"{100 * sum(1 for r in cell if r['exact_status'] == want) / len(cell):>9.0f}%"
                     if cell else f"{'-':>10}")
        print(line)


def section_vs(rows, allocators, ref):
    """Every allocator against one reference, paired on the same task sets.

    One readable column of the all pairs matrix, with the statistics attached:
    a bootstrap interval excluding zero means the difference is real rather
    than sampling luck, and p is a two sided sign test on wins and losses.
    """
    if ref not in allocators:
        return
    by = defaultdict(dict)
    for r in rows:
        by[r["taskset"]][r["allocator"]] = r

    print(f"\n{'=' * 104}\nevery allocator against {ref}\n{'=' * 104}")
    print(f"negative mean difference favours the row, so a negative number "
          f"means it beat {ref}\n")
    print(f"{'allocator':<16}{'n':>7}{'wins':>8}{'tie':>8}{'losses':>8}"
          f"{'mean diff':>12}{'95% CI':>20}{'p':>9}")
    for a in allocators:
        if a == ref:
            continue
        w, t, l, d = paired(by, a, ref)
        if not d:
            continue
        ci = bootstrap_ci(d)
        p = sign_test(w, l)
        print(f"{a:<16}{len(d):>7}{w:>8}{t:>8}{l:>8}{mean(d):>12.1f}"
              f"{(f'[{ci[0]:.1f}, {ci[1]:.1f}]' if ci else '-'):>20}"
              f"{('<0.001' if p < 0.001 else f'{p:.3f}'):>9}")


def section3(rows, allocators, platforms, pair):
    by = defaultdict(dict)
    for r in rows:
        by[r["taskset"]][r["allocator"]] = r

    print(f"\n{'=' * 104}\nleaderboard   (over each allocator's own pool)\n{'=' * 104}")
    print(f"{'allocator':<16}{'n':>7}{'perfect':>10}{'pen hard':>11}{'median':>9}"
          f"{'pen total':>11}{'misses H/F/S':>16}{'cross':>8}")
    for a in allocators:
        cell = [r for r in rows if r["allocator"] == a]
        if not cell:
            continue
        pen = [r["penalty_hard"] for r in cell]
        perf = 100 * mean([float(r["perfect"]) for r in cell])
        print(f"{a:<16}{len(cell):>7}{f'{perf:.0f}%':>10}"
              f"{mean(pen):>11.1f}{median(pen):>9.1f}"
              f"{mean([r['penalty_total'] for r in cell]):>11.1f}"
              f"{sum(r['missed_hard'] for r in cell):>8}"
              f"/{sum(r['missed_firm'] for r in cell)}"
              f"/{sum(r['missed_soft'] for r in cell):<6}"
              f"{mean([r['n_crossings'] for r in cell]):>8.1f}")

    if len(allocators) > 1 and MATRIX:
        print(f"\n{'=' * 104}\nall pairs, on penalty_hard, paired on the same task set"
              f"\n{'=' * 104}")
        print("row wins / tie / column wins,  and the mean difference "
              "(negative favours the row)\n")
        print(f"{'':<12}" + "".join(f"{b:>26}" for b in allocators))
        for a in allocators:
            line = f"{a:<16}"
            for b in allocators:
                if a == b:
                    line += f"{'-':>26}"
                    continue
                w, t, l, d = paired(by, a, b)
                if not d:
                    line += f"{'no overlap':>26}"
                    continue
                line += f"{f'{w}/{t}/{l}':>16}{mean(d):>10.1f}"
            print(line)

    if pair and all(p in allocators for p in pair):
        a, b = pair
        print(f"\n{'=' * 104}\n{a} vs {b}, by axis"
              f"   (on penalty_hard, preference excluded)\n{'=' * 104}")
        print("a 95% bootstrap interval that excludes zero means the difference "
              "is real rather than sampling luck.\np is a two sided sign test on "
              "the wins and losses, ties dropped.\n")
        print(f"{'group':<20}{'n':>6}{a + ' wins':>14}{'tie':>7}{b + ' wins':>14}"
              f"{'mean diff':>12}{'95% CI':>20}{'p':>10}"
              f"   negative favours {a}")

        def block(label, sel):
            w, t, l, d = paired(by, a, b, sel)
            if not d:
                return
            ci = bootstrap_ci(d)
            ci_s = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "-"
            p = sign_test(w, l)
            p_s = "<0.001" if p < 0.001 else f"{p:.3f}"
            print(f"{label:<20}{len(d):>6}{w:>14}{t:>7}{l:>14}{mean(d):>12.1f}"
                  f"{ci_s:>20}{p_s:>10}")

        for p in platforms:
            block(p, lambda r, p=p: r["platform"] == p)
        print()
        for c in sorted({r["coupling"] for r in rows}):
            block(f"coupling {c}", lambda r, c=c: r["coupling"] == c)
        print()
        for l in sorted({r["load"] for r in rows}):
            block(f"load {l}", lambda r, l=l: r["load"] == l)
        print()
        block("ALL", lambda r: True)

    # what the search bought over the allocation it was handed, which estimates
    # how much a one pass packer leaves on the table
    lifts = [(base_of(a), a) for a in allocators
             if base_of(a) and base_of(a) in allocators]
    if lifts:
        print(f"\n{'=' * 104}\nwhat annealing bought over its own starting point"
              f"\n{'=' * 104}")
        print(f"{'start -> search':<20}{'n':>6}{'improved':>10}{'unchanged':>11}"
              f"{'worse':>7}{'pen hard':>20}{'perfect':>18}")
        print(f"{'':<20}{'':>6}{'':>10}{'':>11}{'':>7}"
              f"{'before -> after':>20}{'before -> after':>18}")
        for b, s in lifts:
            w, t, l, d = paired(by, s, b)
            common = [p for p in by.values() if b in p and s in p]
            if not common:
                continue
            pb = mean([p[b]["penalty_hard"] for p in common])
            ps = mean([p[s]["penalty_hard"] for p in common])
            fb = 100 * mean([float(p[b]["perfect"]) for p in common])
            fs = 100 * mean([float(p[s]["perfect"]) for p in common])
            print(f"{b + ' -> ' + s:<20}{len(common):>6}{w:>10}{t:>11}{l:>7}"
                  f"{f'{pb:.1f} -> {ps:.1f}':>20}{f'{fb:.0f}% -> {fs:.0f}%':>18}")

        print(f"\n  by load level, mean penalty_hard")
        print(f"  {'':<18}" + "".join(f"{'load ' + str(l):>22}"
                                      for l in sorted({r['load'] for r in rows})))
        for b, s in lifts:
            line = f"  {b + ' -> ' + s:<18}"
            for l in sorted({r["load"] for r in rows}):
                common = [p for p in by.values() if b in p and s in p
                          and p[b]["load"] == l]
                if not common:
                    line += f"{'-':>22}"
                    continue
                pb = mean([p[b]["penalty_hard"] for p in common])
                ps = mean([p[s]["penalty_hard"] for p in common])
                line += f"{f'{pb:.0f} -> {ps:.0f}':>22}"
            print(line)


def section4(rows, allocators, platforms):
    """Which cells are worth keeping. Drives the tuning loop."""
    # Judged on the two greedy packers only, never on a search or on the learned
    # policy. See the note next to QUALITY_JUDGES in the registry.
    judges = [a for a in QUALITY_JUDGES if a in allocators]
    if len(judges) < 2:
        judges = list(allocators)

    print(f"\n{'=' * 104}\ncell quality: can the allocators be told apart, "
          f"and does the instance fit at all\n{'=' * 104}")
    print(f"  judged on {', '.join(judges)}\n")

    cells = defaultdict(list)
    for r in rows:
        if r["allocator"] in judges:
            cells[(r["platform"], r["coupling"], r["load"])].append(r)

    tally = defaultdict(int)
    verdicts, stats = {}, {}
    for key, cell in cells.items():
        by_ts = defaultdict(dict)
        for r in cell:
            by_ts[r["taskset"]][r["allocator"]] = r["penalty_hard"]

        pairs = [v for v in by_ts.values() if len(v) > 1]
        ties = sum(1 for v in pairs if max(v.values()) - min(v.values()) < 1e-9)
        tie_rate = ties / len(pairs) if pairs else 1.0
        eff = mean([r["load_effective_mean"] for r in cell])
        perfect = mean([float(r["perfect"]) for r in cell])

        v = (TIED if tie_rate > TIE_RATE_WASTED
             else OVER if eff > OVERSUBSCRIBED else USEFUL)
        verdicts[key], stats[key] = v, (tie_rate, eff, perfect)
        tally[v] += 1

    n = len(cells)
    if not n:
        return
    for v in (USEFUL, TIED, OVER):
        print(f"  {v:<16}{tally[v]:>5} of {n} cells  ({100 * tally[v] / n:.0f}%)")
    print(f"\n  tied           allocators score the same on over "
          f"{100 * TIE_RATE_WASTED:.0f}% of the cell, so it separates nothing")
    print(f"  oversubscribed mean effective core load above "
          f"{OVERSUBSCRIBED}, demand exceeds the whole platform")
    print(f"  useful         everything else, including cells where nobody is "
          f"perfect but by different amounts")

    couplings = sorted({r["coupling"] for r in rows})
    loads = sorted({r["load"] for r in rows})
    mark = {TIED: "T", USEFUL: ".", OVER: "O"}
    header = "   ".join(f"c{c}" for c in couplings)
    print(f"\n  load down the side, coupling across.  T tied, O oversubscribed, . useful")
    for p in platforms:
        if not any(k[0] == p for k in verdicts):
            continue
        print(f"\n  {p:<16}{header}")
        for l in loads:
            row = "   ".join(mark[verdicts[(p, c, l)]] for c in couplings
                             if (p, c, l) in verdicts)
            s = [stats[(p, c, l)] for c in couplings if (p, c, l) in verdicts]
            if not s:
                continue
            print(f"    load {l}       {row}      tie "
                  f"{100 * mean([x[0] for x in s]):>3.0f}%  "
                  f"eff {mean([x[1] for x in s]):.2f}  "
                  f"perfect {100 * mean([x[2] for x in s]):>3.0f}%")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", action="store_true",
                    help="the full per (platform, coupling, load) table for every "
                         "allocator. 200 rows each, so a thousand lines with six "
                         "allocators. Pipe it to a file")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), default=None,
                    help="the two allocators to break down by axis. "
                         "Defaults to the two one pass packers.")
    ap.add_argument("--common", action="store_true",
                    help="restrict every section to task sets that every "
                         "allocator was run on")
    ap.add_argument("--eval", action="store_true",
                    help="only task sets the learned policy never trained on. "
                         "Any headline number goes here")
    ap.add_argument("--ref", default="rl",
                    help="the allocator every other one is compared against")
    ap.add_argument("--matrix", action="store_true",
                    help="also print the full all pairs matrix. Eighteen "
                         "columns wide, so off by default")
    ap.add_argument("--no-portfolios", action="store_true",
                    help="leave out best_greedy, best_greedy_sa and oracle")
    ap.add_argument("--axes", nargs="+", default=None,
                    choices=[a for a, _, _ in AXES],
                    help="which groupings the marginals section prints")
    ap.add_argument("--pool", choices=["workload", "hardware"], default=None,
                    help="restrict to unseen task sets on seen platforms, or "
                         "to the five platforms the policy never trained on")
    ap.add_argument("--csv")
    args = ap.parse_args()

    global MATRIX
    MATRIX = args.matrix

    rows = load()
    if not rows:
        raise SystemExit("no allocations. run scripts/run/run_allocators.py first")

    if args.eval:
        rows = [r for r in rows if r["set"] == splits.EVAL]
        if not rows:
            raise SystemExit(
                "no eval allocations. run: python scripts/run/run_allocators.py --eval")
        print("EVAL ONLY: held out task sets, nothing the policy trained on\n")

    if args.pool:
        rows = [r for r in rows if r.get("pool") == args.pool]
        if not rows:
            raise SystemExit(f"no task sets in the {args.pool} pool")
        what = ("unseen task sets on platforms the policy trained on"
                if args.pool == "workload"
                else "the five platforms the policy never trained on")
        print(f"POOL {args.pool}: {what}\n")

    if not args.no_portfolios:
        rows = add_portfolios(rows)

    present = {r["allocator"] for r in rows}
    allocators = order_allocators(present)
    platforms = sorted({r["platform"] for r in rows}, key=pnum)

    section0(rows, allocators)

    if args.common:
        seen = defaultdict(set)
        for r in rows:
            seen[r["taskset"]].add(r["allocator"])
        keep = {t for t, v in seen.items() if len(v) == len(allocators)}
        rows = [r for r in rows if r["taskset"] in keep]
        print(f"\n--common: kept {len(keep)} task sets, {len(rows)} rows")
        if not rows:
            raise SystemExit("no task set was run by every allocator")

    pair = args.pair
    if pair is None:
        bases = [a for a in allocators if not a.startswith("sa_")]
        pair = bases[:2] if len(bases) >= 2 else allocators[:2]

    if args.cells:
        section1(rows, allocators, platforms)
    section2(rows, allocators, args.axes)
    section_chains(rows, allocators)
    section_time(rows, allocators, platforms)
    section_budget(rows, allocators)
    section3(rows, allocators, platforms, pair)
    section_vs(rows, allocators, args.ref)
    section_optimal(rows, allocators)
    section4(rows, allocators, platforms)

    if args.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
