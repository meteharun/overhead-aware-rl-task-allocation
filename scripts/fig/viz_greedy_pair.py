#!/usr/bin/env python3
"""Best fit and worst fit on the same six tasks, side by side.

    python scripts/fig/viz_greedy_pair.py

Writes out/writing/greedy_pair.png.

Best fit and worst fit on the same six tasks, combined into one
figure. The task queue is identical in both, so it is drawn once on the left and
the two results follow it: best fit, then worst fit.

Showing them together is the point. The two rules fill the cores differently and
still leave the same task over, so what fails is the rule family rather than the
tie-break inside it. As two figures a reader has to hold one page in their head
to see that.

Drawn with figparts' own primitives, so this figure, the two stills and the
animation cannot diverge. Nothing in figparts is modified: each panel is its
own axes over the same 0..100 space, cropped to the part of it that panel uses.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import figparts as gv
import paths

# the queue runs from QX to its widest label, the cores from the "Rejected"
# label to the right of "N free". Panel widths follow those spans, so a unit is
# the same length in all three and the bars keep one scale across the figure
QUEUE_X = (0.0, 42.0)
CORE_X = (37.0, 102.0)
# one x unit to one y unit, at (10, 5.8)
ASPECT = 1.724

DPI = 300
PAD = 18
OUTDIR = "writing"
PANELS = [("bfd", "Best fit"), ("wfd", "Worst fit")]


def draw_queue(ax):
    """The six tasks in the order they arrive, not the order they are placed.

    Unsorted. Both rules sort by size before packing, and showing
    the sorted list here would hide that step rather than illustrate it.
    """
    ax.text(gv.QX, 83, "Tasks", fontsize=12, fontweight="bold", color=gv.MUTED)
    for name, w in gv.TASKS:
        y = gv.slot_y(gv.unsorted_slot[name])
        gv.box(ax, gv.QX, y, w * gv.UNIT, gv.BAR_H, gv.COLORS[name], ec=gv.INK,
               lw=1.4, z=4)
        ax.text(gv.QX + w * gv.UNIT + 1.2, y + gv.BAR_H / 2, f"{name} · {w}",
                ha="left", va="center", fontsize=11, color=gv.INK)


def draw_result(ax, policy, title):
    """The cores, the rule that filled them, and what would not fit.

    `draw_tracks` and `draw_tasks` put the rejected task and its label hard
    against the left edge of the track, which leaves it hanging off the side of
    the panel. They are handed a state with nothing rejected so they skip it,
    and it is drawn below centred on the track instead. Everything still goes
    through figparts' own primitives.
    """
    placed, rejected = gv.greedy(policy)
    st = gv.state([], placed, rejected, gv.unsorted_slot)

    ax.text(gv.BX, 83, "Cores", fontsize=12, fontweight="bold", color=gv.MUTED)
    # the rule, in ink just above the first track, so it reads as a heading for
    # the cores under it rather than as another axis label
    ax.text(gv.BX, 72.4, title, fontsize=13, fontweight="bold", color=gv.INK)

    gv.draw_tracks(ax, dict(st, rejected=[]))
    gv.draw_tasks(ax, dict(st, rejected=[]))

    mid = gv.BX + gv.BIN_W / 2
    for name, w in rejected:
        bw = w * gv.UNIT
        x = mid - bw / 2
        gv.box(ax, x, gv.REJ_Y, bw, gv.BAR_H, "#F6DADA", ec=gv.BAD, lw=1.8,
               hatch="///", z=4)
        ax.text(x - 2.5, gv.REJ_Y + gv.BAR_H / 2, "Rejected", ha="right",
                va="center", fontsize=12, fontweight="bold", color=gv.BAD)
        ax.text(x + bw + 1.5, gv.REJ_Y + gv.BAR_H / 2, f"{name} · {w}",
                ha="left", va="center", fontsize=11, color=gv.BAD,
                fontweight="bold")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--dir", default=OUTDIR)
    ap.add_argument("--name", default="greedy_pair")
    ap.add_argument("--width", type=float, default=14.0,
                    help="figure width in inches. Height follows from it, so "
                         "the bars keep the proportions of the single stills")
    args = ap.parse_args()

    spans = [QUEUE_X[1] - QUEUE_X[0]] + [CORE_X[1] - CORE_X[0]] * len(PANELS)
    units = sum(spans)
    height = args.width / units * 100.0 / ASPECT

    fig, axes = plt.subplots(1, 1 + len(PANELS), figsize=(args.width, height),
                             dpi=args.dpi,
                             gridspec_kw={"width_ratios": spans, "wspace": 0.0})
    fig.patch.set_facecolor("white")
    for ax, xlim in zip(axes, [QUEUE_X] + [CORE_X] * len(PANELS)):
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 100)
        ax.axis("off")

    draw_queue(axes[0])
    for ax, (policy, title) in zip(axes[1:], PANELS):
        draw_result(ax, policy, title)

    fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
    save(fig, paths.out(args.dir, args.name))
    plt.close(fig)

    for policy, title in PANELS:
        placed, rejected = gv.greedy(policy)
        loads = {c: sum(w for _, w in placed[c]) for c in gv.CORES}
        print(f"{title:<10}" + "  ".join(f"{c} {loads[c]:>3}/{gv.CAP}"
                                         for c in gv.CORES)
              + "   rejected " + ", ".join(n for n, _ in rejected))


if __name__ == "__main__":
    main()
