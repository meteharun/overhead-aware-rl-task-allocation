"""What a search budget buys, from each starting point.

    python scripts/run/run_allocators.py --only sa_rl sa_bfd ... --eval
    python scripts/fig/viz_budget.py
    python scripts/fig/viz_budget.py --panels penalty
    python scripts/fig/viz_budget.py --panels perfect

Writes out/writing/budget_curve.png.

The curve is read straight off the annealed runs in data/allocations. Every
annealed allocator writes down its best so far as it passes each budget, so one
full run already answers "what would this have produced with B evaluations" for
every B on the x axis, with no separate sweep that could disagree with the
headline table.

Every line is the same search with the same move set and the same schedule. Only
the allocation it started from differs, so the gap between two lines at a given
budget is what the starting point was worth.

Penalty is on a log axis because it spans two orders of magnitude, from about
400 at budget zero to under ten at four thousand. On a linear axis everything
below fifty is a flat line along the bottom and the interesting part of the
story is invisible. `--scale neg` gives the negated symlog variant instead.

The x axis is categorical rather than logarithmic, because the budgets are not
evenly spaced and one of them is zero.

Colour carries one thing only: which allocation the search started from. RL is
the subject, so it is the one saturated hue and the one heavy line. The
heuristics are grouped by family, best fit in amber, worst fit in violet,
earliest finish time in teal, with the criticality ordered variant of each one
step darker and dashed.

The right panel carries a reference line at the share of held out task sets that
some method has already produced a zero penalty allocation for. That share is
known to be reachable, since the evaluator scored those allocations. Whether the
rest are reachable is unknown rather than settled, so the line is a floor on
what is achievable and not a cap, and a better allocator could pass it. It is
drawn because a panel whose top is 100 quietly implies the remaining gap is
failure, which is the one thing the data does not say. `--no-ceiling` removes it.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import figstyle
from palette import HAIRLINE, INK, INK_MUTED, SURFACE

ROOT = Path(__file__).resolve().parent.parent.parent
ALLOC = ROOT / "data" / "allocations"
OUT = ROOT / "out" / "writing" / "budget_curve.png"

# Fallback for the reference line, used only when no allocation files are on
# disk to count. `solved_ceiling` replaces it with the measured share wherever
# it can, which is the share already shown to be reachable rather than an
# estimate of how many task sets are solvable.
CEILING = 85.0

# "neg" plots the negated penalty so every curve climbs towards zero. "log"
# plots the penalty itself on a log axis, falling.
SCALE = "log"

# The negated panel is symmetric log, not linear. Penalties run from about 400
# down to about 5, so a linear axis spends its whole height on the left hand end
# and leaves the four thousand evaluation results sitting on top of each other
# in the last one percent. Symlog is linear inside this band and logarithmic
# outside it, which keeps zero at the top and a real gap at the far end.
# Set it to None for a plain linear axis.
LINTHRESH = 10.0
TICKS = [0, 5, 10, 25, 50, 100, 200, 400, 800]

# The subject first. Its hue is the only saturated one on the chart. Colours,
# widths and names live in figstyle so this chart and the repair charts cannot
# drift apart.
SERIES = ["rl", "bfd", "bfd_crit", "wfd", "wfd_crit", "heft", "heft_crit"]

# Outputs of a deepening run, not starting points. Their curves run far past
# 4,000 evaluations, and a search given a hundred times the budget does not
# belong on a chart whose whole subject is what each starting point was worth at
# the same budget. viz_budget_deep.py draws those.
DEEP_SUFFIX = "_deep"


def load(alloc_dir=ALLOC, eval_only=True):
    """The budget curve, aggregated out of the annealed allocation files.

    Returns budgets, one series per starting point, and the mean penalty and
    perfect rate at each budget.

    Series are named by the allocator the search started from, so `sa_bfd` on
    disk becomes `bfd` on the chart, since the question is what the starting
    point was worth rather than how the search did.
    """
    pen, prf, seen = {}, {}, {}
    for f in sorted(alloc_dir.glob("sa_*.jsonl")):
        if f.stem.endswith(DEEP_SUFFIX):
            continue
        start = f.stem[3:]
        rows = 0
        hard, perfect = {}, {}
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                if eval_only and r.get("set") not in (None, "eval"):
                    continue
                curve = r["summary"].get("sa_curve")
                if not curve:
                    continue
                rows += 1
                for b, v in curve.items():
                    hard.setdefault(b, []).append(v["hard"])
                    perfect.setdefault(b, []).append(float(v["perfect"]))
        if not rows:
            continue
        seen[start] = rows
        pen[start] = {b: sum(v) / len(v) for b, v in hard.items()}
        prf[start] = {b: sum(v) / len(v) for b, v in perfect.items()}

    if not pen:
        raise SystemExit(
            f"no search curves in {alloc_dir}. Run the annealed allocators:\n"
            "  python scripts/run/run_allocators.py --only sa_rl sa_bfd --eval")

    # Only budgets every series carries. The chart compares starting points at
    # one budget, so a budget one of them never reached has nothing to compare,
    # and indexing a series at it would simply fail.
    common = set.intersection(*({int(b) for b in d} for d in pen.values()))
    dropped = {int(b) for d in pen.values() for b in d} - common
    if dropped:
        print(f"ignoring {len(dropped)} budget(s) not present in every series: "
              + ", ".join(f"{b:,}" for b in sorted(dropped)))
    return {"n": max(seen.values()), "budgets": sorted(common),
            "starts": list(pen), "penalty": pen, "perfect": prf}


def solved_ceiling(alloc_dir=ALLOC, default=CEILING, eval_only=True):
    """The share of task sets some method has produced a zero penalty
    allocation for.

    Any allocation scoring zero on penalty_hard is a feasibility witness, since
    the timing evaluator checked that allocation directly. So the claim rests
    on the evaluator rather than on the branch and bound solver's pruning.

    Counted as the union over every allocator's file rather than from the
    solver's own `schedulable` count, which is lower: the solver opens from
    `sa_rl` and the six one pass heuristics and never sees the annealed
    heuristics, so instances those had already solved come back as timeouts.
    The solver's certificates are a strict subset of this.

    Instances nothing has solved are unknown rather than unsolvable, so this is
    a lower bound.
    """
    seen, ok = set(), set()
    for f in sorted(alloc_dir.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                if eval_only and r.get("set") != "eval":
                    continue
                seen.add(r["taskset_id"])
                if r["summary"]["penalty_hard"] == 0:
                    ok.add(r["taskset_id"])
    if not seen:
        return default, None
    return 100 * len(ok) / len(seen), len(seen)


def crossing(budgets, ys, target):
    """Where a curve first reaches `target`, interpolated between checkpoints.

    The checkpoints double and then double again, so naming the first one that
    clears the target overstates what the search needed. The first heuristic is
    under RL's one pass result at the 100 evaluation checkpoint, but it was
    already under it at about 74. Reporting 100 credits the heuristic with a
    third more search than it actually used.

    Interpolation is log budget against log penalty, the same reading the budget
    equivalence table in `analyse_allocations` uses. Returns (budget, x), with x
    the fractional position on the categorical axis, or None if the curve never
    reaches the target.
    """
    for i in range(1, len(budgets)):
        if ys[i] > target:
            continue
        lo, hi, p_lo, p_hi = budgets[i - 1], budgets[i], ys[i - 1], ys[i]
        # nothing to interpolate across: already under it at the previous
        # checkpoint, or a zero budget whose logarithm does not exist
        if p_lo <= target or lo <= 0 or min(p_lo, p_hi) <= 0:
            return hi, float(i)
        f = ((math.log(p_lo) - math.log(target))
             / (math.log(p_lo) - math.log(p_hi)))
        f = min(max(f, 0.0), 1.0)
        b = math.exp(math.log(lo) + f * (math.log(hi) - math.log(lo)))
        return b, (i - 1) + f
    return None


def draw(d, out, scale=SCALE, panels=("penalty", "perfect"), ceiling=None):
    budgets = d["budgets"]
    starts = [s for s in SERIES if s in d["starts"]]
    starts += [s for s in d["starts"] if s not in starts]
    x = list(range(len(budgets)))
    neg = scale == "neg"
    sign = -1.0 if neg else 1.0      # one place decides which way is up

    wide = 6.9 * len(panels) if len(panels) > 1 else 8.6
    fig, axes = plt.subplots(1, len(panels), figsize=(wide, 5.2), dpi=150,
                             gridspec_kw={"wspace": 0.22})
    axes = list(axes) if len(panels) > 1 else [axes]
    fig.patch.set_facecolor(SURFACE)
    panel = dict(zip(panels, axes))

    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(color=HAIRLINE, linewidth=0.6, alpha=0.7)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(HAIRLINE)
        a.tick_params(colors=INK_MUTED, labelsize=9)
        a.set_xticks(x)
        a.set_xticklabels([str(b) for b in budgets], fontsize=9)
        a.set_xlabel("Search budget (timing model evaluations)", fontsize=10,
                     color=INK_MUTED)

    worst = 0.0
    for name in starts:
        col, lw, dash, ms, z = figstyle.style(name)
        series = {"penalty": [sign * d["penalty"][name][str(b)] for b in budgets],
                  "perfect": [100 * d["perfect"][name][str(b)] for b in budgets]}
        worst = max(worst, max(abs(v) for v in series["penalty"]))
        for key, a in panel.items():
            a.plot(x, series[key], color=col, linewidth=lw, linestyle=dash,
                   marker="o", markersize=ms, markeredgecolor=SURFACE,
                   markeredgewidth=1.4, zorder=z)

    if "penalty" in panel:
        ax = panel["penalty"]
        # What the policy hands over before any search, and the budget the first
        # heuristic needs to reach it. The gap between the two is the whole
        # claim of the one pass comparison, stated on the search's own axis.
        if "rl" in starts and not neg:
            target = d["penalty"]["rl"][str(budgets[0])]
            col = figstyle.style("rl")[0]
            ax.axhline(target, color=col, lw=1.1, linestyle=(0, (4, 3)),
                       alpha=0.85, zorder=2)
            hits = [crossing(budgets,
                             [d["penalty"][s][str(b)] for b in budgets], target)
                    for s in starts if s != "rl"]
            hits = [h for h in hits if h]
            if hits:
                # the earliest budget any heuristic needed, not the first
                # checkpoint at which one of them happened to be measured under
                b_star, xpos = min(hits)
                ax.plot([xpos], [target], marker="o", markersize=7,
                        markerfacecolor="none", markeredgecolor=INK,
                        markeredgewidth=1.6, zorder=6)
                ax.annotate(f"no heuristic reaches RL's one-pass result\n"
                            f"below about {b_star:,.0f} evaluations",
                            xy=(xpos, target), xytext=(12, 20),
                            textcoords="offset points", ha="left", va="bottom",
                            fontsize=9, color=INK,
                            arrowprops=dict(arrowstyle="-", color=INK_MUTED,
                                            lw=0.9), zorder=6)
        if neg:
            if LINTHRESH:
                ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=1.0)
                ax.set_ylim(-worst * 1.15, LINTHRESH * 0.35)
                # symlog names its own ticks in powers of ten, which reads badly
                # on a penalty axis
                ticks = [-t for t in TICKS if t <= worst * 1.15]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{int(t)}" for t in ticks], fontsize=9)
            else:
                ax.set_ylim(-worst * 1.06, worst * 0.05)
            ax.set_ylabel("Negated mean penalty", fontsize=10, color=INK_MUTED)
            ax.axhline(0.0, color=INK_MUTED, linewidth=1.2, zorder=2)
        else:
            ax.set_yscale("log")
            ax.set_ylabel("Penalty (log scale)", fontsize=10, color=INK_MUTED)
        ax.set_title("Mean penalty", fontsize=12, color=INK, loc="left",
                     fontweight="bold", pad=10)

    if "perfect" in panel:
        bx = panel["perfect"]
        bx.set_ylabel("Instances with zero penalty (%)", fontsize=10,
                      color=INK_MUTED)
        bx.set_ylim(0, 100)
        bx.set_title("Perfect rate", fontsize=12, color=INK, loc="left",
                     fontweight="bold", pad=10)
        if ceiling:
            bx.axhline(ceiling, color=INK, linewidth=1.4, linestyle=(0, (6, 3)),
                       zorder=2)
            # not "proved". A zero penalty allocation on record witnesses that
            # the instance is solvable, and nothing here proves the rest are not
            bx.annotate(f"{ceiling:.1f}% proved to be feasible",
                        xy=(len(budgets) - 1, ceiling), xytext=(0, 5),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=9, color=INK)

    handles = []
    for n in starts:
        col, lw, dash, ms, _ = figstyle.style(n)
        handles.append(Line2D([], [], color=col, lw=lw, linestyle=dash,
                              marker="o", markersize=ms,
                              markeredgecolor=SURFACE, markeredgewidth=1.2,
                              label=figstyle.short(n)))
    fig.legend(handles=handles, loc="lower center", ncol=len(starts),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))

    fig.subplots_adjust(top=0.92, bottom=0.20, left=0.075, right=0.985)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panels", nargs="+", choices=["penalty", "perfect"],
                    default=["penalty", "perfect"],
                    help="which panels to draw. One of them alone gives a "
                         "single chart at full width")
    ap.add_argument("--scale", choices=["log", "neg"], default=SCALE,
                    help="log falls towards zero, neg plots the negated penalty "
                         "and climbs")
    ap.add_argument("--ceiling", type=float, default=None,
                    help="override the witnessed feasible share")
    ap.add_argument("--no-ceiling", action="store_true",
                    help="drop the ceiling line from the perfect rate panel")
    ap.add_argument("--linthresh", type=float, default=LINTHRESH,
                    help="neg only. Penalty band the axis stays linear inside")
    ap.add_argument("--all", action="store_true",
                    help="include training task sets. Headline numbers do not")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    d = load(eval_only=not args.all)
    witnessed, n = solved_ceiling(eval_only=not args.all)
    if n:
        print(f"ceiling: {witnessed:.1f}% witnessed feasible over {n:,} "
              f"instances, a zero penalty allocation on record from some method")
    top = None if args.no_ceiling else (args.ceiling or witnessed)
    LINTHRESH = args.linthresh or None
    print(f"{d['n']:,} task sets, budgets {d['budgets']}")
    draw(d, Path(args.out), args.scale, tuple(args.panels), top)
