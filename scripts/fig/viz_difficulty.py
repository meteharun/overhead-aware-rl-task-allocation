#!/usr/bin/env python3
"""Quality against the three axes the dataset varies.

    python scripts/fig/viz_difficulty.py
    python scripts/fig/viz_difficulty.py --axes load devices

Writes out/writing/offline-difficulty.png.

Small multiples: one column per axis, the same allocators and the same colours
throughout. Three separate figures would make the reader carry a scale between
pages, and the point of the figure is the comparison between axes, not any one
of them.

All three panels share one y scale. That is the whole reason the grid works:
load is the only axis that moves the result much, and coupling and devices look
flat precisely because they are, next to it. Giving each panel its own scale
would manufacture three equally dramatic trends out of one real one.

Every allocator here has been refined: the one-pass allocation followed by 4,000
candidate evaluations. Portfolio is the per-instance best of the six refined
heuristics, so it is six searches rather than one.

Load is the calibrated difficulty axis and behaves like one. Coupling is not
monotone for every allocator, which is a result rather than a defect and is
worth saying out loud. Devices is the axis the policy was built for, and the
single device column is the one place a bin packer beats it.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
# analyse_allocations does the loading and aggregation these figures draw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import figstyle as F
from figstyle import MUTED
from palette import SURFACE

FIGSIZE = (13.2, 4.6)
DPI = 300
PAD = 18
OUTDIR = "writing"

AXES = {"load": "Load level", "coupling": "Coupling level",
        "devices": "Devices on the platform"}
# every refined allocator, each family beside its criticality variant, with the
# policy last so the legend ends on the subject
BARS = ["sa_bfd", "sa_bfd_crit", "sa_wfd", "sa_wfd_crit",
        "sa_heft", "sa_heft_crit", "sa_rl"]


def gather(names, axes):
    """Mean penalty per allocator at each level of each axis.

    No portfolio here. It is the best of six searches rather than one, so it
    belongs to a different cost class and putting it on the same axes invites
    the comparison the figure is not making.
    """
    import analyse_allocations as A
    rows = [r for r in A.load() if r["set"] == "eval"]
    out, levels = {}, {}
    for axis in axes:
        levels[axis] = sorted({r[axis] for r in rows if axis in r})
        for a in names:
            for v in levels[axis]:
                cell = [r for r in rows
                        if r["allocator"] == a and r.get(axis) == v]
                if cell:
                    out[(a, axis, v)] = (
                        sum(r["penalty_hard"] for r in cell) / len(cell),
                        len(cell))
    return out, levels


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
    ap.add_argument("--name", default="offline-difficulty")
    ap.add_argument("--axes", nargs="+", choices=list(AXES),
                    default=["load", "coupling", "devices"])
    args = ap.parse_args()

    import paths
    names = BARS
    stats, levels = gather(names, args.axes)

    fig, axes = plt.subplots(1, len(args.axes), figsize=FIGSIZE,
                             dpi=args.dpi, squeeze=False,
                             gridspec_kw={"wspace": 0.13})
    fig.patch.set_facecolor(SURFACE)
    F.chrome(*axes.ravel())

    # one scale across all three, so the axes can be compared with each other
    top = max(v[0] for v in stats.values())
    for c, axis in enumerate(args.axes):
        ax = axes[0][c]
        xs = list(range(len(levels[axis])))
        for a in names:
            ys = [stats[(a, axis, v)][0] for v in levels[axis]
                  if (a, axis, v) in stats]
            if len(ys) != len(xs):
                continue
            col, lw, dash, ms, z = F.style(a)
            ax.plot(xs, ys, color=col, lw=lw, linestyle=dash, marker="o",
                    markersize=ms, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, zorder=z)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in levels[axis]], fontsize=9.5)
        ax.set_xlim(-0.25, len(xs) - 0.75)
        ax.set_ylim(0, top * 1.08)
        ax.set_xlabel(AXES[axis], fontsize=10.5, color=MUTED)
        if c == 0:
            ax.set_ylabel("Mean penalty", fontsize=10.5, color=MUTED)
        F.title(ax, AXES[axis])

    handles = []
    for a in names:
        col, lw, dash, ms, _ = F.style(a)
        handles.append(Line2D([], [], color=col, lw=lw, linestyle=dash,
                              marker="o", markersize=ms,
                              markeredgecolor=SURFACE, markeredgewidth=1.2,
                              label=F.short(a)))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))

    fig.subplots_adjust(top=0.90, bottom=0.235, left=0.058, right=0.99)
    save(fig, paths.out(args.dir, args.name))
    plt.close(fig)

    for axis in args.axes:
        print(f"\n{AXES[axis]}: {levels[axis]}")
        for a in names:
            got = [stats.get((a, axis, v)) for v in levels[axis]]
            if any(g is None for g in got):
                continue
            print(f"  {F.short(a):<12}"
                  + "".join(f"{g[0]:>9.1f}" for g in got))


if __name__ == "__main__":
    main()
