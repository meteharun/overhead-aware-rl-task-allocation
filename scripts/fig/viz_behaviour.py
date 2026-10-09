#!/usr/bin/env python3
"""Miss composition and penalty composition of the refined allocators.

    python scripts/fig/viz_behaviour.py

Writes out/writing/offline-behaviour.png.

Penalty says whether an allocator is good. This says what it gave up to get
there, which is the only way to tell a policy that found a better trade from one
that merely searched harder.

Every allocator is refined: a constructive pass followed by 4,000 candidate
evaluations. Nothing one pass appears here.

Left, which tasks miss. Bar length is the total count over the pool, so the
comparison between allocators is the number of failures. The ratios beside each
bar are its split across the three criticality bands, in the legend's order, and
they are nearly identical everywhere: search removes almost every hard miss
whatever it started from, so what separates the allocators is how many misses
are left rather than which kind.

Right, what the remaining penalty is made of. penalty_hard is two different
failures added together, tasks that missed a deadline and chains that ran past
their budget, and the sum hides that the ratio moves.

Both panels are horizontal and share their rows, so an allocator sits on the
same line in each and the two can be read across.
"""

import argparse
import sys
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
# analyse_allocations does the loading and aggregation these figures draw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import figstyle as F
from figstyle import CHAIN_PEN, CRIT, INK, MISS_PEN, MUTED
from palette import SURFACE, ink_on

# the grid runs under every label, so each one carries its own ground
BOX = dict(boxstyle="round,pad=0.18", facecolor=SURFACE, edgecolor="none",
           alpha=0.92)

FIGSIZE = (12.6, 4.6)
DPI = 300
PAD = 18
OUTDIR = "writing"

SERIES = ["sa_bfd", "sa_bfd_crit", "sa_wfd", "sa_wfd_crit",
          "sa_heft", "sa_heft_crit", "sa_rl"]
CRIT_ORDER = [("hard", "HARD"), ("firm", "FIRM"), ("soft", "SOFT")]
BAR_H = 0.62


def gather():
    import analyse_allocations as A
    rows = [r for r in A.load() if r["set"] == "eval"]
    out = {}
    for a in SERIES:
        cell = [r for r in rows if r["allocator"] == a]
        if not cell:
            continue
        n = len(cell)
        out[a] = {
            "miss": sum(r["penalty_misses"] for r in cell) / n,
            "budget": sum(r["penalty_budgets"] for r in cell) / n,
            "hard": sum(r["missed_hard"] for r in cell),
            "firm": sum(r["missed_firm"] for r in cell),
            "soft": sum(r["missed_soft"] for r in cell),
            "n": n,
        }
    return out


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


def misses(ax, stats, names, ys):
    top = max(sum(stats[n][k] for k, _ in CRIT_ORDER) for n in names)
    for y, n in zip(ys, names):
        left = 0.0
        for key, _ in CRIT_ORDER:
            ax.barh(y, stats[n][key], left=left, height=BAR_H, color=CRIT[key],
                    edgecolor=INK, lw=0.9, zorder=3)
            left += stats[n][key]
        # The hard band is under one percent of the bar, so a label inside it
        # would be a label on two pixels. The split goes beside the bar instead,
        # in the legend's order, which needs no leader lines and cannot collide.
        share = " / ".join(f"{100 * stats[n][k] / left:.1f}"
                           for k, _ in CRIT_ORDER)
        ax.text(left + top * 0.025, y, f"{left:,.0f}      {share} %",
                ha="left", va="center", fontsize=9, color=INK, zorder=5,
                bbox=BOX)
    ax.set_xlim(0, top * 1.52)
    ax.set_xlabel("Missed tasks, evaluation pool total", fontsize=10,
                  color=MUTED)
    F.title(ax, "Deadline misses by criticality")
    ax.legend(handles=[Patch(facecolor=CRIT[k], edgecolor=INK, lw=0.9,
                             label=lab) for k, lab in CRIT_ORDER],
              loc="lower right", frameon=False, fontsize=9.5,
              handlelength=1.3, borderaxespad=0.8)


def components(ax, stats, names, ys):
    top = max(stats[n]["miss"] + stats[n]["budget"] for n in names)
    for y, n in zip(ys, names):
        s = stats[n]
        tot = s["miss"] + s["budget"]
        left = 0.0
        # the share goes inside its own segment, where it needs no ground of its
        # own and cannot be read against the wrong bar
        for v, col in ((s["miss"], MISS_PEN), (s["budget"], CHAIN_PEN)):
            ax.barh(y, v, left=left, height=BAR_H, color=col, edgecolor=INK,
                    lw=0.9, zorder=3)
            ax.text(left + v / 2, y, f"{100 * v / tot:.0f}%", ha="center",
                    va="center", fontsize=9, fontweight="bold",
                    color=ink_on(col), zorder=4)
            left += v
        ax.text(tot + top * 0.035, y, f"{tot:.1f}", ha="left", va="center",
                fontsize=9, color=INK, zorder=5, bbox=BOX)
    ax.set_xlim(0, top * 1.22)
    ax.set_xlabel("Mean penalty", fontsize=10,
                  color=MUTED)
    F.title(ax, "Penalty composition")
    ax.legend(handles=[Patch(facecolor=MISS_PEN, edgecolor=INK, lw=0.9,
                             label="Missed deadlines"),
                       Patch(facecolor=CHAIN_PEN, edgecolor=INK, lw=0.9,
                             label="Chain budget overruns")],
              loc="lower right", frameon=False, fontsize=9.5,
              handlelength=1.3, borderaxespad=0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--dir", default=OUTDIR)
    ap.add_argument("--name", default="offline-behaviour")
    args = ap.parse_args()

    import paths
    stats = gather()
    names = [n for n in SERIES if n in stats]
    # first in the list sits at the top, which is how a list is read
    ys = list(range(len(names)))[::-1]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=args.dpi,
                                 gridspec_kw={"wspace": 0.06,
                                              "width_ratios": [1.28, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    F.chrome(ax, bx)

    misses(ax, stats, names, ys)
    components(bx, stats, names, ys)

    for a, labelled in ((ax, True), (bx, False)):
        a.set_yticks(ys)
        a.set_yticklabels([F.short(n) for n in names] if labelled else [],
                          fontsize=10)
        a.set_ylim(-0.8, len(names) - 0.5)
        a.tick_params(axis="y", length=0)

    fig.subplots_adjust(top=0.90, bottom=0.145, left=0.082, right=0.995)
    save(fig, paths.out(args.dir, args.name))
    plt.close(fig)

    print(f"\n{stats[names[0]]['n']:,} task sets")
    print(f"{'alloc':<12}{'misses':>8}{'HARD%':>7}{'FIRM%':>7}{'SOFT%':>7}"
          f"{'penalty':>9}{'chain%':>8}")
    for n in names:
        s = stats[n]
        tot = sum(s[k] for k, _ in CRIT_ORDER)
        pen = s["miss"] + s["budget"]
        print(f"{F.short(n):<12}{tot:>8,}"
              + "".join(f"{100 * s[k] / tot:>7.1f}" for k, _ in CRIT_ORDER)
              + f"{pen:>9.1f}{100 * s['budget'] / pen:>7.0f}%")


if __name__ == "__main__":
    main()
