#!/usr/bin/env python3
"""One swap is enough: annealing rescuing the worst-fit allocation.

    python scripts/fig/viz_anneal_example.py

Writes out/writing/anneal_example.png.

Continues the greedy example. Worst fit finishes with 90 and 95
and cannot place the last task of size 15, because the ten and five units it has
left are on different cores. Swapping T6 and T3 moves five units from one core to
the other, which changes neither core's total by much and opens a gap of exactly
the right size. That is the move a one pass packer cannot make, since it would
have to reconsider a decision it has already committed to.

Two panels. The first repeats the worst-fit result with the proposed swap
marked. The second is where that swap leads, with the leftover task placed and
both cores exactly full.

Drawn with figparts' primitives, so it matches the greedy figures and the
animation. Nothing in figparts is modified.
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

FIGSIZE = (16.2, 4.4)
DPI = 300
PAD = 18
OUTDIR = "writing"

SWAP = ("T6", "T3")
YLIM = (18.0, 88.0)
# right edge sits clear of the "n free" labels, which start at 93.5. The panels
# are then sized so both hold ten data units per inch, the same scale as the
# greedy figures, which is what stops the labels growing relative to the bars
FULL_X = (2.0, 105.0)               # panel with the task list
CORES_X = (46.0, 105.0)             # panel showing the cores alone

# worst fit, then the swap, then the task that was left over
WFD = {"C1": [("T2", 65), ("T6", 25)],
       "C2": [("T4", 45), ("T1", 30), ("T3", 20)]}
FILLED = {"C1": [("T2", 65), ("T3", 20), ("T5", 15)],
          "C2": [("T4", 45), ("T1", 30), ("T6", 25)]}

# before and after. The intermediate state, swapped but with T5 still waiting,
# adds a panel without adding an idea
PANELS = [(WFD, [("T5", 15)], True, SWAP, FULL_X, "Cores (after WFD)"),
          (FILLED, [], False, None, CORES_X, "Cores (after SA)")]


def draw_queue(ax):
    ax.text(gv.QX, 83, "Tasks", fontsize=12, fontweight="bold", color=gv.MUTED)
    for name, w in gv.TASKS:
        y = gv.slot_y(gv.unsorted_slot[name])
        gv.box(ax, gv.QX, y, w * gv.UNIT, gv.BAR_H, gv.COLORS[name], ec=gv.INK,
               lw=1.4, z=4)
        ax.text(gv.QX + w * gv.UNIT + 1.2, y + gv.BAR_H / 2, f"{name} · {w}",
                ha="left", va="center", fontsize=11, color=gv.INK)


def draw_rejected(ax, rejected):
    """The leftover task, right aligned under the end of the tracks.

    figparts puts it at the left edge, which is where a queue belongs. Here it
    reads better under the gaps it failed to fit into, and its label then lines
    up with the free space labels above it.
    """
    total = sum(w * gv.UNIT for _, w in rejected) + 2.0 * (len(rejected) - 1)
    x = gv.BX + (gv.BIN_W - total) / 2.0        # centred under the tracks
    for name, w in rejected:
        wide = w * gv.UNIT
        gv.box(ax, x, gv.REJ_Y, wide, gv.BAR_H, "#F6DADA", ec=gv.BAD, lw=1.8,
               hatch="///", z=4)
        ax.text(x + wide + 1.5, gv.REJ_Y + gv.BAR_H / 2, f"{name} · {w}",
                ha="left", va="center", fontsize=11, color=gv.BAD,
                fontweight="bold")
        x += wide + 2.0
    # label sits with its bar rather than in the C1 / C2 column, so the two read
    # as one thing
    left = gv.BX + (gv.BIN_W - total) / 2.0
    ax.text(left - 1.5, gv.REJ_Y + gv.BAR_H / 2, "Rejected", ha="right",
            va="center", fontsize=12, fontweight="bold", color=gv.BAD)


def draw_swap(ax, pos, a, b):
    """A two headed arrow between the pair the move would exchange."""
    xa, ya = pos[a]
    xb, yb = pos[b]
    ca = xa + gv.W[a] * gv.UNIT / 2
    cb = xb + gv.W[b] * gv.UNIT / 2
    top, bot = (ya, yb) if ya > yb else (yb, ya)
    ax.annotate("", xy=(cb, bot + gv.BAR_H + 0.6), xytext=(ca, top - 0.6),
                arrowprops=dict(arrowstyle="<->", color=gv.ACCENT, lw=2.2),
                zorder=9)
    ax.text((ca + cb) / 2 + 1.6, (top + bot + gv.BAR_H) / 2, "swap", ha="left",
            va="center", fontsize=11, color=gv.ACCENT, fontweight="bold",
            zorder=9)


def strip_numbers(ax, min_x=40.0):
    """Names only on the bars inside the cores. The list on the left keeps its
    sizes, since that is where a reader looks them up.

    Two forms to catch. A wide bar carries `T3 - 20` inside it. A bar between
    3.2 and 7 units carries the name inside and floats the size above it as a
    bare number, which is what T5 does once it is placed.
    """
    short = {f"{n} · {w}": n for n, w in gv.TASKS}
    bare = {str(w) for _, w in gv.TASKS}
    for t in list(ax.texts):
        if t.get_position()[0] <= min_x:
            continue
        label = t.get_text()
        if label in short:
            t.set_text(short[label])
        elif label in bare:
            t.remove()


def draw(ax, placed, rejected, queue, swap, xlim, heading):
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    # rejected kept out of the state so figparts draws neither the bar nor
    # the label, and draw_rejected can place them itself
    st = gv.state([], placed, [], gv.unsorted_slot)
    if queue:
        draw_queue(ax)
    ax.text(gv.BX, 83, heading, fontsize=12, fontweight="bold", color=gv.MUTED)
    gv.draw_tracks(ax, st)
    gv.draw_tasks(ax, st)
    strip_numbers(ax)
    if rejected:
        draw_rejected(ax, rejected)
    if swap:
        draw_swap(ax, gv.positions(st), *swap)


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
    args = ap.parse_args()

    widths = [(x1 - x0) for *_, (x0, x1), _ in PANELS]
    fig, axes = plt.subplots(1, len(PANELS), figsize=FIGSIZE, dpi=args.dpi,
                             gridspec_kw={"width_ratios": widths, "wspace": 0.04})
    fig.patch.set_facecolor("white")
    for ax, args_ in zip(axes, PANELS):
        draw(ax, *args_)
    save(fig, paths.out(args.dir, "anneal_example"))
    plt.close(fig)


if __name__ == "__main__":
    main()
