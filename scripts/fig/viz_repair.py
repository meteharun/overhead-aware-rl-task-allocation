#!/usr/bin/env python3
"""The two figures of the core-failure chapter.

    python scripts/fig/viz_repair.py

Writes into out/writing:

  repair-endpoints        repair against a refined rebuild, on three measures
  repair-rebuild-budget   how much search a rebuild needs to overtake repair

Both read out/runs/repair.jsonl, streamed, because it holds 409,600 reactions.

**Endpoints.** Three panels, because quality, hit rate and disruption are three
different quantities and a height in one must not be read as a height in
another. Hue is the allocator, as everywhere else; lightness is the reaction,
pale for repair and solid for rebuild. The ten-evaluation follow-up is left
out, being unrestricted and able to move pinned survivors, so it belongs to
neither endpoint.

**Budget.** Quality against the time the reaction actually took, which is the
axis a recovery requirement is written in. Search budget is on a second axis
above it, since that is the knob rather than the quantity, and the two are not
proportional: a rebuild pays for its constructive pass before the first
evaluation.

Putting time on the axis is what lets repair and rebuild share a panel. At
"budget zero" they are two different constructions and cannot sit at the same
x; in time they are two points, and repair turns out both cheaper and better
than an unsearched rebuild. Repair is therefore a marker, not a horizontal
line.

A rebuild's time is its constructive pass plus its search, reconstructed as the
recorded total minus the search recorded at the last checkpoint, so it is
comparable with a repair time rather than covering search alone.

BFD and WFD repair to within 0.3 penalty of each other in 0.68 ms each, so their
markers land on top of one another. Coincident repair markers are nudged apart
along the time axis by up to four percent, which is a presentation offset and
not a measurement. Nothing else on the figure is moved.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import figstyle as F
from figstyle import INK, MUTED, NEUTRAL, NEUTRAL_PALE, PALE
from palette import SURFACE

ROOT = Path(__file__).resolve().parent.parent.parent
ROWS = ROOT / "out" / "runs" / "repair.jsonl"

DPI = 300
PAD = 18
OUTDIR = "writing"

# heuristics first, the policy last, as in the offline figures
METHODS = ["bfd", "wfd", "heft", "rl"]
BOX = dict(boxstyle="round,pad=0.18", facecolor=SURFACE, edgecolor="none",
           alpha=0.92)


def load(path):
    """Mean penalty, perfect rate, migrations and time at each endpoint, plus
    the rebuild's own budget curve."""
    acc = defaultdict(lambda: {"pen": 0.0, "perf": 0, "mig": 0.0, "ms": [],
                               "spen": 0.0, "sms": [], "smig": 0.0, "n": 0})
    # The first reaction an allocator computes carries its one-off setup. For
    # `rl` that is loading the checkpoint and the first forward pass, and it
    # measured 14,982 ms against a median of 2.8, which on its own moved the
    # mean repair time from 2.94 to 4.11. `run_allocators` preloads to keep this
    # out of its timings and `repair.py` does not, so it is dropped here.
    warmed = set()
    curve = defaultdict(lambda: defaultdict(
        lambda: {"hard": 0.0, "perf": 0, "moved": 0.0, "ms": 0.0, "n": 0}))
    before, tasksets = 0.0, set()
    n = 0
    with Path(path).open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["allocator"] not in warmed:
                warmed.add(r["allocator"])
                continue
            if r["freed"] not in ("0", "rebuild"):
                continue
            a = acc[(r["allocator"], r["freed"])]
            a["pen"] += r["penalty_after"]
            a["perf"] += r["penalty_after"] <= 1e-9
            a["mig"] += r["migrations"]
            a["ms"].append(r["ms"])
            # the same reaction after its ten evaluation follow-up
            a["spen"] += r["searched_penalty"]
            a["sms"].append(r["searched_ms"])
            a["smig"] += r["searched_migrations"]
            a["n"] += 1
            before += r["penalty_before"]
            tasksets.add(r["taskset"])
            n += 1
            for b, v in (r.get("sa_curve") or {}).items():
                c = curve[r["allocator"]][int(b)]
                c["hard"] += v["hard"]
                c["perf"] += bool(v["perfect"])
                c["moved"] += v.get("moved", 0)
                c["ms"] += v["ms"]
                c["n"] += 1
    if not acc:
        raise SystemExit(f"no usable rows in {path}. run scripts/run/repair.py")

    stats = {k: {"pen": v["pen"] / v["n"],
                 "perf": 100 * v["perf"] / v["n"],
                 "mig": v["mig"] / v["n"],
                 "ms": statistics.median(v["ms"]),
                 "ms_mean": sum(v["ms"]) / len(v["ms"]),
                 "spen": v["spen"] / v["n"],
                 "smig": v["smig"] / v["n"],
                 "sms_mean": sum(v["sms"]) / len(v["sms"]),
                 "n": v["n"]} for k, v in acc.items()}
    curves = {a: {b: {"hard": c["hard"] / c["n"],
                      "perf": 100 * c["perf"] / c["n"],
                      "moved": c["moved"] / c["n"],
                      "ms": c["ms"] / c["n"]}
                  for b, c in sorted(d.items())} for a, d in curve.items()}
    return stats, curves, before / n, len(tasksets)


def save(fig, path, pad=PAD):
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    flat = Image.new("RGB", im.size, im.getpixel((0, 0)))
    bbox = (ImageChops.difference(im, flat).convert("L")
            .point(lambda v: 255 if v > 6 else 0).getbbox())
    x0, y0, x1, y1 = bbox
    im.crop((max(0, x0 - pad), max(0, y0 - pad),
             min(im.size[0], x1 + pad), min(im.size[1], y1 + pad))).save(path)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
def tradeoff(stats, out, dpi):
    """The trade itself: what each reaction costs in time, and what it buys.

    Back to back, sharing rows, with the allocator names down the middle. Time
    grows to the left and penalty to the right, so each allocator draws a pair
    of opposed bars and the crossing between the two panels is the trade. Repair
    is a stub on the left and a long bar on the right. Rebuild is the reverse.

    Time is on a log axis because the two reactions are three orders of
    magnitude apart, 0.8 ms against 350, and on a linear axis every repair bar
    would be a quarter of one percent of the panel. Bars are drawn from the
    axis minimum rather than from zero, which has no place on a log scale.

    Earlier drafts gave time its own axis in a scatter and then reduced it to a
    text label in a dumbbell. Neither let the times be compared by eye, which is
    half of the comparison the chapter is making.
    """
    fig, (tx, px) = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=dpi,
                                 gridspec_kw={"wspace": 0.085,
                                              "width_ratios": [1.0, 1.05]})
    fig.patch.set_facecolor(SURFACE)
    F.chrome(tx, px)
    for a in (tx, px):
        a.grid(axis="y", visible=False)

    ys = list(range(len(METHODS)))[::-1]
    height, off = 0.34, 0.185
    reactions = (("rebuild", 1), ("0", -1))

    t_lo = 0.30
    t_hi = max(stats[(m, "rebuild")]["ms_mean"] for m in METHODS) * 3.4
    for y, m in zip(ys, METHODS):
        col = F.style(m)[0]
        pale = PALE.get(m, NEUTRAL_PALE)
        for freed, sign in reactions:
            v = stats[(m, freed)]["ms_mean"]
            face = col if freed == "rebuild" else pale
            # from the axis minimum, not from zero, which does not exist here
            tx.barh(y + sign * off, v - t_lo, left=t_lo, height=height,
                    color=face, edgecolor=INK, lw=0.9, zorder=3)
            tx.text(v * 1.22, y + sign * off,
                    f"{v:,.1f}" if v < 10 else f"{v:,.0f}", ha="left",
                    va="center", fontsize=9, color=INK, zorder=4)

            q = stats[(m, freed)]["pen"]
            px.barh(y + sign * off, q, height=height, color=face,
                    edgecolor=INK, lw=0.9, zorder=3)
            px.text(q + 3.5, y + sign * off, f"{q:.1f}", ha="left",
                    va="center", fontsize=9, color=INK, zorder=4)

    tx.set_xscale("log")
    tx.set_xlim(t_hi, t_lo)                       # grows leftwards
    ticks = [t for t in (0.5, 1, 5, 10, 50, 100, 500, 1000) if t_lo < t < t_hi]
    tx.set_xticks(ticks)
    tx.set_xticklabels([f"{t:g}" for t in ticks], fontsize=9.5)
    tx.minorticks_off()
    tx.set_xlabel("Mean reaction time (ms, log scale)", fontsize=10.5,
                  color=MUTED)
    F.title(tx, "What it costs")
    tx.set_yticks([])

    px.set_xlim(0, max(stats[(m, "0")]["pen"] for m in METHODS) * 1.22)
    px.set_xlabel("Mean penalty", fontsize=10.5, color=MUTED)
    F.title(px, "What it buys")
    px.set_yticks(ys)
    px.set_yticklabels([F.short(m) for m in METHODS], fontsize=10.5)
    px.tick_params(axis="y", length=0, pad=8)

    for a in (tx, px):
        a.set_ylim(-0.62, len(METHODS) - 0.32)

    fig.legend(handles=[Patch(facecolor=NEUTRAL_PALE, edgecolor=INK, lw=0.9,
                              label="Repair, orphans only"),
                        Patch(facecolor=NEUTRAL, edgecolor=INK, lw=0.9,
                              label="Rebuild, 1,000 evaluations")],
               loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(top=0.90, bottom=0.205, left=0.015, right=0.985)
    save(fig, out)


PANELS = {"pen": ("Mean penalty", "{:.1f}"),
          "perf": ("Perfect rate (%)", "{:.0f}"),
          # every method moves 2.5 assignments repairing and about 34
          # rebuilding, so this panel draws the same two bars four times and
          # separates nothing. Off by default, kept for `--endpoint-panels`
          "mig": ("Changed assignments", "{:.1f}")}


def endpoints(stats, out, dpi, keys=("pen", "perf")):
    panels = [(k, *PANELS[k]) for k in keys]
    wide = 4.4 * len(panels) + 0.6
    fig, axes = plt.subplots(1, len(panels), figsize=(wide, 4.4), dpi=dpi,
                             squeeze=False, gridspec_kw={"wspace": 0.2})
    axes = axes[0]
    fig.patch.set_facecolor(SURFACE)
    F.chrome(*axes)

    width = 0.34
    for ax, (key, title, fmt) in zip(axes, panels):
        top = max(stats[(m, f)][key] for m in METHODS for f in ("0", "rebuild"))
        for i, m in enumerate(METHODS):
            col = F.style(m)[0]
            for j, (freed, face) in enumerate((("0", PALE.get(m, NEUTRAL_PALE)),
                                               ("rebuild", col))):
                v = stats[(m, freed)][key]
                x = i + (j - 0.5) * width
                ax.bar(x, v, width=width * 0.9, color=face, edgecolor=INK,
                       lw=1.0, zorder=3)
                ax.text(x, v + top * 0.025, fmt.format(v), ha="center",
                        va="bottom", fontsize=8.5, color=INK, zorder=4,
                        bbox=BOX)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([F.short(m) for m in METHODS], fontsize=10)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_ylim(0, top * 1.18)
        ax.tick_params(axis="x", length=0)
        F.title(ax, title)

    fig.legend(handles=[Patch(facecolor=NEUTRAL_PALE, edgecolor=INK, lw=1.0,
                              label="Repair, orphans only"),
                        Patch(facecolor=NEUTRAL, edgecolor=INK, lw=1.0,
                              label="Rebuild, 1,000 evaluations")],
               loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(top=0.91, bottom=0.185,
                        left=0.055 * 3 / len(panels), right=0.99)
    save(fig, out)


# --------------------------------------------------------------------------
def crossover(budgets, ys, target):
    """Budget at which a falling curve first passes under its own reference."""
    for (b0, v0), (b1, v1) in zip(zip(budgets, ys), zip(budgets[1:], ys[1:])):
        if v0 > target >= v1 and b1 > 0:
            if v0 == v1:
                return b1
            f = (v0 - target) / (v0 - v1)
            lo = b0 if b0 > 0 else 1.0
            return lo * (b1 / lo) ** f
    return None


def repair_curve(path):
    """The budget curve of the search that followed each repair.

    Written by scripts/run/repair_search.py, which replays the same failures and
    anneals the orphans-only repair for the same budget a rebuild gets. Returns
    {method: {budget: {...}}} and the mean construction time, or (None, None)
    when the file is not there, so the figure still draws without it.
    """
    path = Path(path)
    if not path.exists():
        return None, None
    acc = defaultdict(lambda: defaultdict(
        lambda: {"hard": 0.0, "perf": 0, "moved": 0.0, "ms": 0.0, "n": 0}))
    build = defaultdict(lambda: [0.0, 0])
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r["allocator"]
            build[m][0] += r["ms"]
            build[m][1] += 1
            for b, v in (r.get("sa_curve") or {}).items():
                c = acc[m][int(b)]
                c["hard"] += v["hard"]
                c["perf"] += bool(v["perfect"])
                c["moved"] += v.get("moved", 0)
                c["ms"] += v["ms"]
                c["n"] += 1
    if not acc:
        return None, None
    curves = {m: {b: {"hard": c["hard"] / c["n"],
                      "perf": 100 * c["perf"] / c["n"],
                      "moved": c["moved"] / c["n"],
                      "ms": c["ms"] / c["n"]}
                  for b, c in sorted(d.items())} for m, d in acc.items()}
    return curves, {m: t / n for m, (t, n) in build.items()}


def spread(xs, gap=0.045):
    """Nudge coincident points apart along a logarithmic axis.

    Two markers at the same place are one marker. This pushes a coincident
    group symmetrically outwards in log space, by a few percent, so both are
    visible. Presentation only, and small enough that no value moves by more
    than the width of its own marker.
    """
    import math
    out = dict(xs)
    order = sorted(xs, key=lambda k: xs[k])

    runs, run = [], [order[0]]
    for k in order[1:]:
        if abs(math.log10(xs[k]) - math.log10(xs[run[-1]])) < gap:
            run.append(k)
        else:
            runs.append(run)
            run = [k]
    runs.append(run)

    for run in runs:
        if len(run) < 2:
            continue
        mid = (len(run) - 1) / 2
        for i, k in enumerate(run):
            out[k] = xs[k] * 10 ** ((i - mid) * gap)
    return out


def _curve(ax, xs, ys, col, label, repair):
    """One budget curve. Repair is dashed and hollow, rebuild solid and
    filled, in both panels, so the same allocation is drawn the same way
    wherever it appears."""
    if repair:
        ax.plot(xs, ys, color=col, lw=2.2, linestyle=(0, (5, 2)), marker="o",
                markersize=5.8, markerfacecolor=SURFACE, markeredgecolor=col,
                markeredgewidth=1.6, zorder=7, label=label)
    else:
        ax.plot(xs, ys, color=col, lw=2.2, marker="o", markersize=5.8,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=6,
                label=label)


def budget(stats, curves, out, dpi, rcurves=None, rbuild=None, focus="rl"):
    """What a search budget buys after a core failure.

    Left, every allocator rebuilding, with the policy's repair curve laid over
    them as the reference to beat. Right, every allocator repairing. The policy's
    repair therefore appears in both panels, drawn identically, which is what
    lets the left panel be read as a comparison rather than as a second figure.

    The panels share the time axis but not the penalty axis. Sharing both put
    the repair curves in the bottom third of their box, against a scale set by
    the worst rebuild, and nothing in them could be read. The right panel is
    scaled to its own data instead, and its axis maximum is stated so the two
    are not mistaken for one scale.

    Time only, on one axis. Budget and time are not in a fixed relationship,
    since an evaluation runs slower on a fuller core, so carrying both was one
    axis more than the figure could justify.
    """
    budgets = sorted(next(iter(curves.values())))
    last = budgets[-1]

    # a rebuild's recorded time covers its constructive pass and its search, and
    # the curve records the search alone, so the difference is the construction
    build = {m: stats[(m, "rebuild")]["ms_mean"] - curves[m][last]["ms"]
             for m in METHODS}
    pair = bool(rcurves and rbuild)

    def reb(m, b):
        return build[m] + curves[m][b]["ms"]

    def rep(m, b):
        return rbuild[m] + rcurves[m][b]["ms"]

    fig, axes = plt.subplots(1, 2 if pair else 1,
                             figsize=(13.6 if pair else 9.8, 5.7), dpi=dpi,
                             squeeze=False, sharex=True,
                             gridspec_kw={"wspace": 0.115})
    axes = list(axes[0])
    fig.patch.set_facecolor(SURFACE)
    F.chrome(*axes)
    for a in axes:
        a.set_xscale("log")

    for m in METHODS:
        _curve(axes[0], [reb(m, b) for b in budgets],
               [curves[m][b]["hard"] for b in budgets],
               F.style(m)[0], F.short(m), repair=False)
    if pair:
        rb = sorted(rcurves[focus])
        _curve(axes[0], [rep(focus, b) for b in rb],
               [rcurves[focus][b]["hard"] for b in rb],
               F.style(focus)[0], f"{F.short(focus)}, repair", repair=True)
        for m in METHODS:
            if m in rcurves:
                rb = sorted(rcurves[m])
                _curve(axes[1], [rep(m, b) for b in rb],
                       [rcurves[m][b]["hard"] for b in rb],
                       F.style(m)[0], F.short(m), repair=True)

    titles = ["Rebuild" + (f", against {F.short(focus)} repair" if pair else ""),
              "Repair"]
    for a, t in zip(axes, titles):
        a.legend(loc="upper right", frameon=False, fontsize=9.5,
                 handlelength=2.0, borderaxespad=0.7)
        F.title(a, t)

    everything = [reb(m, b) for m in METHODS for b in budgets]
    if pair:
        everything += [rep(m, b) for m in rcurves for b in rcurves[m]]
    lo, hi = min(everything), max(everything)
    axes[0].set_xlim(lo * 0.62, hi * 1.55)
    # each panel to its own penalty range. The rebuild scale is three times the
    # repair one, and forcing the second into the first hides everything in it
    axes[0].set_ylim(0, max(curves[m][0]["hard"] for m in METHODS) * 1.06)
    if pair:
        axes[1].set_ylim(0, max(rcurves[m][0]["hard"] for m in rcurves) * 1.06)

    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500)
             if lo * 0.62 <= t <= hi * 1.55]
    for a in axes:
        a.set_xticks(ticks)
        a.set_xticklabels([f"{t:g}" for t in ticks], fontsize=9.5)
        a.minorticks_off()
        a.set_xlabel("Mean reaction time (ms, log scale)", fontsize=10.5,
                     color=MUTED)
        a.set_ylabel("Mean penalty", fontsize=10.5, color=MUTED)
        a.tick_params(labelleft=True)

    fig.subplots_adjust(top=0.905, bottom=0.115, left=0.062, right=0.99)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--dir", default=OUTDIR)
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--name", default="repair-rebuild-budget-mean",
                    help="basename of the budget figure. The -mean suffix "
                         "marks the version whose x axis is mean reaction "
                         "time throughout, so it cannot be confused with the "
                         "earlier file that mixed a median with a mean")
    ap.add_argument("--only", choices=("budget", "endpoints", "tradeoff"),
                    default=None, help="draw just this one")
    ap.add_argument("--repair-curve",
                    default=str(ROOT / "out" / "runs" / "repair_search.jsonl"),
                    help="budget curve of the search after each repair, from "
                         "scripts/run/repair_search.py. Drawn when present")
    ap.add_argument("--endpoint-name", default="repair-endpoints-quality",
                    help="basename of the endpoints figure")
    ap.add_argument("--endpoint-panels", nargs="+", default=["pen", "perf"],
                    choices=list(PANELS),
                    help="which panels the endpoints figure carries")
    args = ap.parse_args()

    import paths
    if not Path(args.rows).exists():
        raise SystemExit(f"no {args.rows}. run scripts/run/repair.py first")
    stats, curves, before, n_ts = load(args.rows)

    if args.only is None or args.only == "endpoints":
        endpoints(stats, paths.out(args.dir, args.endpoint_name), args.dpi,
                  tuple(args.endpoint_panels))
    if args.only is None or args.only == "tradeoff":
        tradeoff(stats, paths.out(args.dir, "repair-endpoints-time"), args.dpi)
    if args.only is None or args.only == "budget":
        rcurves, rbuild = repair_curve(args.repair_curve)
        if rcurves:
            print(f"repair search curve: {args.repair_curve}")
        budget(stats, curves, paths.out(args.dir, args.name), args.dpi,
               rcurves, rbuild)

    fails = stats[("rl", "0")]["n"]
    print(f"\n{fails:,} failures over {n_ts:,} task sets, "
          f"penalty before failure {before:.2f}\n")
    # both statistics, because the figure plots the mean and the text tables
    # quote the median, and the two are 47 percent apart on RL repair
    print(f"{'method':<8}{'repair pen':>11}{'perf':>7}{'moved':>7}"
          f"{'mean ms':>9}{'med ms':>8}"
          f"{'rebuild pen':>13}{'perf':>7}{'moved':>7}{'mean ms':>9}{'med ms':>8}")
    for m in METHODS:
        a, b = stats[(m, "0")], stats[(m, "rebuild")]
        print(f"{F.short(m):<8}{a['pen']:>11.1f}{a['perf']:>6.1f}%"
              f"{a['mig']:>7.1f}{a['ms_mean']:>9.2f}{a['ms']:>8.2f}"
              f"{b['pen']:>13.1f}{b['perf']:>6.1f}%{b['mig']:>7.1f}"
              f"{b['ms_mean']:>9.0f}{b['ms']:>8.0f}")

    budgets = sorted(next(iter(curves.values())))
    print(f"\ncrossover, interpolated between checkpoints")
    for m in METHODS:
        pen = crossover(budgets, [curves[m][b]["hard"] for b in budgets],
                        stats[(m, "0")]["pen"])
        prf = next((b for b in budgets
                    if curves[m][b]["perf"] >= stats[(m, "0")]["perf"]), None)
        print(f"  {F.short(m):<6}penalty at {pen:>6.0f} evaluations"
              f"    perfect rate at {prf if prf else '>1000'}")


if __name__ == "__main__":
    main()
