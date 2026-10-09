#!/usr/bin/env python3
"""What repair and rebuild do to an allocation, on one worked instance.

    python scripts/fig/viz_repair_concept.py

Writes out/writing/repair-example.png.

The thesis version of the slide in viz_repair_example.py, and it shares that
file's scenario so the two cannot drift apart. What it drops is everything about
quality: no deadlines, no late blocks, no penalty, no milliseconds. Those are
measured over 12,800 failures in the budget figure, and repeating them here on a
hand authored instance of six tasks invites the reader to take them as evidence.
This figure defines the two responses and nothing else.

Three cores, six tasks, one failure.

  1  the working allocation
  2  repair    C3 dies. Its two tasks go in the gaps that already exist on the
               surviving cores and every other task stays where it was
  3  rebuild   every task is placed again on the two cores that are left, so
               tasks that were never stranded also move

A heavy outline marks a task that sits on a different core from panel 1, which
is the only difference between the two responses and the reason repair is cheap.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import figstyle as F
import palette as P
from figstyle import INK, MUTED
from palette import SURFACE, ink_on
from viz_repair_example import KERNEL, REBUILD, REPAIR, TASKS, TCOL, WORKING

FIGSIZE = (12.2, 3.6)
DPI = 300
PAD = 18
OUTDIR = "writing"

CORES = ["C1", "C2", "C3"]
LANE_H = 0.74
LANE_Y = {"C1": 2.0, "C2": 1.0, "C3": 0.0}
LANE_FILL = {"C1": P.CPU_RAMP["Kryo_Gen6"][0], "C2": P.CPU_RAMP["Kryo_Gen6"][1],
             "C3": P.UNKNOWN_RAMP[0]}
DEAD_FILL = "#f1ecec"

# thin and muted where a task stayed, heavy ink where it moved. Weight alone was
# too quiet a difference at this size, so the two also differ in colour
EDGE_STAY = (MUTED, 1.0)
EDGE_MOVE = (INK, 2.6)

PANELS = [("Working allocation", WORKING, None),
          ("Repair", REPAIR, WORKING),
          ("Rebuild", REBUILD, WORKING)]


def schedule(plan):
    """Lay each core's list out end to end, kernel then task."""
    blocks = []
    for core, names in plan.items():
        t = 0.0
        for name in names:
            blocks.append(dict(core=core, kind="kernel", t=t, d=KERNEL))
            t += KERNEL
            dur = TASKS[name][1]
            blocks.append(dict(core=core, kind="run", t=t, d=dur, task=name))
            t += dur
    return blocks


def horizon():
    """The widest core over all three panels, so the lanes are not mostly
    empty and the same instant sits at the same x in every panel."""
    ends = [b["t"] + b["d"]
            for _, plan, _ in PANELS for b in schedule(plan)]
    return max(ends) * 1.06


def elsewhere(plan, base):
    """Tasks sitting on a different core from the working allocation."""
    if base is None:
        return set()
    where = {n: c for c, ns in base.items() for n in ns}
    return {n for c, ns in plan.items() for n in ns if where[n] != c}


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


def panel(ax, title, plan, base, span, dead, ylabels):
    F.chrome(ax)
    ax.grid(False)
    ax.set_xlim(-1.0, span)
    ax.set_ylim(-0.30, 2.98)
    ax.set_xticks([])
    ax.set_yticks([LANE_Y[c] + LANE_H / 2 for c in CORES])
    ax.set_yticklabels(CORES if ylabels else [], fontsize=10,
                       fontweight="bold", color=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    for core in CORES:
        y = LANE_Y[core]
        gone = dead and core == "C3"
        ax.add_patch(Rectangle((0, y), span, LANE_H,
                               facecolor=DEAD_FILL if gone else LANE_FILL[core],
                               edgecolor=P.HAIRLINE, lw=1.0, zorder=2))
        if gone:
            ax.plot([0, span], [y, y + LANE_H], color=F.BAD_X, lw=1.2, zorder=4)
            ax.plot([0, span], [y + LANE_H, y], color=F.BAD_X, lw=1.2, zorder=4)
            ax.text(span / 2, y + LANE_H / 2, "core failed", ha="center",
                    va="center", fontsize=9.5, color=F.BAD_X, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor=SURFACE,
                              edgecolor="none"))

    moved = elsewhere(plan, base)
    for b in schedule(plan):
        y = LANE_Y[b["core"]]
        if b["kind"] == "kernel":
            ax.add_patch(Rectangle((b["t"], y), b["d"], LANE_H,
                                   facecolor="none", edgecolor=P.GAP, lw=1.0,
                                   hatch="////", zorder=5))
            continue
        fill = TCOL[b["task"]]
        col, lw = EDGE_MOVE if b["task"] in moved else EDGE_STAY
        ax.add_patch(Rectangle((b["t"], y), b["d"], LANE_H, facecolor=fill,
                               edgecolor=col, lw=lw, zorder=5))
        ax.text(b["t"] + b["d"] / 2, y + LANE_H / 2, b["task"], ha="center",
                va="center", fontsize=9, fontweight="bold",
                color=ink_on(fill), zorder=7)

    F.title(ax, title)
    return moved


def legend(fig):
    handles = [Patch(facecolor=TCOL[n], edgecolor=MUTED, lw=1.0, label=n)
               for n in TASKS]
    handles.append(Patch(facecolor="none", edgecolor=P.GAP, lw=1.0, hatch="////",
                         label="Kernel overhead"))
    handles.append(Patch(facecolor=SURFACE, edgecolor=INK, lw=EDGE_MOVE[1],
                         label="Reassigned"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9.5, handlelength=1.4, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.02))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--dir", default=OUTDIR)
    ap.add_argument("--name", default="repair-example")
    args = ap.parse_args()

    import paths
    span = horizon()

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=args.dpi,
                             gridspec_kw={"wspace": 0.22})
    fig.patch.set_facecolor(SURFACE)
    moved = [panel(ax, title, plan, base, span, dead=i >= 1, ylabels=i == 0)
             for i, (ax, (title, plan, base)) in enumerate(zip(axes, PANELS))]
    legend(fig)

    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.035, right=0.995)
    save(fig, paths.out(args.dir, args.name))
    plt.close(fig)

    for (title, plan, _), m in zip(PANELS, moved):
        print(f"{title:<20}{len(m)} reassigned   {sorted(m)}")


if __name__ == "__main__":
    main()
