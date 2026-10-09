"""The search budget curve, carried far past the 4,000 the thesis compares at.

    python scripts/run/deepen_all.py --evals 500000
    python scripts/run/deepen_to_allocations.py
    python scripts/fig/viz_budget_deep.py

Writes out/writing/budget_curve_deep.png.

This is `viz_budget.py` with every line allowed to run on. Each start stops at
4,000 candidate evaluations, where the thesis leaves it, and then continues to
500,000 on the instances it had not solved by then.

The chart is about what happens to the gaps. At 4,000 the starting point
decides the result. Carry the same search a hundred times further and the four
curves converge, which says the learned start buys search budget rather than a
quality the heuristics cannot reach.

The curve is recorded rather than reconstructed: the recorder sits in the model
wrapper during the deepened run, so it sees every evaluation and snapshots at a
fixed ladder of budgets, with no unsampled span and no interpolation.

An instance that reached zero stopped there, and holding its zero out to larger
budgets is exact since the best-so-far cannot get worse. An instance that spent
its whole budget holds its last value.

Everything else follows `viz_budget.py`: colour carries the starting point,
penalty is on a log axis, and the x axis is categorical because the budgets are
not evenly spaced and one of them is zero.
"""

import argparse
import json
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
OUT = ROOT / "out" / "writing" / "budget_curve_deep.png"

BASE_BUDGET = 4000        # where the thesis run ends and the deepening starts
DEEP_SUFFIX = "_deep"
SERIES = ["rl", "bfd", "bfd_crit", "wfd", "wfd_crit", "heft", "heft_crit"]


def load(deep_file, eval_only=True):
    """Budget curves per starting point, allowing series of different lengths.

    `viz_budget.load` assumes every series carries every budget. Here one
    series is longer, so the curves come back as {budget: value} and the
    drawing code plots each over the budgets it actually has.
    """
    files = sorted(ALLOC.glob("sa_*.jsonl"))
    pen, prf, seen = {}, {}, {}
    for f in files:
        if not f.exists():
            continue
        start = f.stem[3:]
        hard, perfect, rows = {}, {}, 0
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
                    hard.setdefault(int(b), []).append(v["hard"])
                    perfect.setdefault(int(b), []).append(float(v["perfect"]))
        if not rows:
            continue
        seen[start] = rows
        pen[start] = {b: sum(v) / len(v) for b, v in hard.items()}
        prf[start] = {b: sum(v) / len(v) for b, v in perfect.items()}

    if not any(s.endswith(DEEP_SUFFIX) for s in pen):
        raise SystemExit(
            "no deepened series found. Run:\n"
            "  python scripts/run/deepen_all.py --evals 500000\n"
            "  python scripts/run/deepen_to_allocations.py")
    return {"n": max(seen.values()),
            "budgets": sorted({b for d in pen.values() for b in d}),
            "starts": list(pen), "penalty": pen, "perfect": prf}


def witnessed(deep_file, eval_only=True):
    """Share of task sets some method has a zero timing penalty allocation for."""
    files = sorted(ALLOC.glob("*.jsonl"))
    if deep_file and deep_file.resolve() not in {f.resolve() for f in files}:
        files.append(deep_file)
    seen, ok = set(), set()
    for f in files:
        if not f.exists():
            continue
        with f.open() as fh:
            for line in fh:
                r = json.loads(line)
                if eval_only and r.get("set") != "eval":
                    continue
                seen.add(r["taskset_id"])
                if r["summary"]["penalty_hard"] == 0:
                    ok.add(r["taskset_id"])
    return (100 * len(ok) / len(seen), len(seen)) if seen else (None, 0)


def tick(b):
    """Budget as a short label. The deepened ladder starts at 4,100, 4,250 and
    4,500, so integer thousands would print three ticks in a row as 4k."""
    if b < 1000:
        return f"{b:,}"
    if b < 10_000:
        return f"{b / 1000:g}k"
    return f"{b // 1000}k"


def draw(d, out, panels=("penalty", "perfect"), ceiling=None):
    budgets = d["budgets"]
    pos = {b: i for i, b in enumerate(budgets)}
    starts = [s for s in SERIES if s in d["starts"]]
    starts += [s for s in d["starts"] if s not in starts]

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
        a.set_xticks(list(pos.values()))
        a.set_xticklabels([tick(b) for b in budgets], fontsize=8.5,
                          rotation=45, ha="right")
        a.set_xlabel("Search budget (timing model evaluations)", fontsize=10,
                     color=INK_MUTED)
        # where the thesis run stops and the deepening takes over
        a.axvline(pos[BASE_BUDGET], color=INK_MUTED, lw=1.0,
                  linestyle=(0, (2, 3)), zorder=1)
        a.annotate("heuristics stop here", xy=(pos[BASE_BUDGET], 1.005),
                   xycoords=("data", "axes fraction"), ha="right", va="bottom",
                   fontsize=8, color=INK_MUTED)

    for name in starts:
        deep = name.endswith(DEEP_SUFFIX)
        col, lw, dash, ms, z = figstyle.style(
            name[:-len(DEEP_SUFFIX)] if deep else name)
        bs = sorted(d["penalty"][name])
        if deep:
            # the short run already draws everything up to 4,000
            bs = [b for b in bs if b >= BASE_BUDGET]
        xs = [pos[b] for b in bs]
        for key, a in panel.items():
            ys = [(100 * d[key][name][b]) if key == "perfect" else d[key][name][b]
                  for b in bs]
            a.plot(xs, ys, color=col, linewidth=lw, linestyle=dash, marker="o",
                   markersize=ms, markeredgecolor=SURFACE, markeredgewidth=1.4,
                   zorder=z + (2 if deep else 0))

    if "penalty" in panel:
        ax = panel["penalty"]
        ax.set_yscale("log")
        ax.set_ylabel("Mean timing penalty (log scale)", fontsize=10,
                      color=INK_MUTED)
        ax.set_title("Mean penalty", fontsize=12, color=INK, loc="left",
                     fontweight="bold", pad=10)
        deep = [s for s in starts if s.endswith(DEEP_SUFFIX)]
        subject = "rl" + DEEP_SUFFIX
        subject = subject if subject in deep else deep[0]
        end = max(d["penalty"][subject])
        lo = min(d["penalty"][s][end] for s in deep)
        hi = max(d["penalty"][s][end] for s in deep)
        ax.annotate(f"every start converges to {lo:.2f}-{hi:.2f}\n"
                    f"at {end // 1000:,}k evaluations",
                    xy=(pos[end], d["penalty"][subject][end]),
                    xytext=(0.26, 0.17), textcoords="axes fraction",
                    ha="left", va="center", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9,
                                    shrinkB=6),
                    zorder=7)

    if "perfect" in panel:
        bx = panel["perfect"]
        bx.set_ylabel("Instances with zero timing penalty (%)", fontsize=10,
                      color=INK_MUTED)
        bx.set_ylim(0, 100)
        bx.set_title("Perfect rate", fontsize=12, color=INK, loc="left",
                     fontweight="bold", pad=10)
        if ceiling:
            bx.axhline(ceiling, color=INK, linewidth=1.4, linestyle=(0, (6, 3)),
                       zorder=2)
            bx.annotate(f"{ceiling:.1f}% witnessed feasible",
                        xy=(len(budgets) - 1, ceiling), xytext=(0, 5),
                        textcoords="offset points", ha="right", va="bottom",
                        fontsize=9, color=INK)

    handles = []
    for n in starts:
        if n.endswith(DEEP_SUFFIX):
            continue
        col, lw, dash, ms, _ = figstyle.style(n)
        handles.append(Line2D([], [], color=col, lw=lw, linestyle=dash,
                              marker="o", markersize=ms,
                              markeredgecolor=SURFACE, markeredgewidth=1.2,
                              label=figstyle.short(n)))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))

    fig.subplots_adjust(top=0.92, bottom=0.26, left=0.075, right=0.985)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep", default=str(ALLOC / "sa_rl_deep.jsonl"),
                    help="the converted deepened file, wherever it was written")
    ap.add_argument("--panels", nargs="+", choices=["penalty", "perfect"],
                    default=["penalty", "perfect"])
    ap.add_argument("--no-ceiling", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="include training task sets. Headline numbers do not")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    deep = Path(args.deep)
    d = load(deep, eval_only=not args.all)
    top, n = witnessed(deep, eval_only=not args.all)
    if n:
        print(f"ceiling: {top:.1f}% witnessed feasible over {n:,} instances")
    print(f"{d['n']:,} task sets, budgets {d['budgets']}")
    for s in sorted(d["penalty"]):
        e = max(d["penalty"][s])
        print(f"  {s:<12} ends at {e:>9,}: penalty {d['penalty'][s][e]:>7.3f}, "
              f"perfect {100 * d['perfect'][s][e]:>5.1f}%")
    draw(d, Path(args.out), tuple(args.panels),
         None if args.no_ceiling else top)
