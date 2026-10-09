#!/usr/bin/env python3
"""Does the policy transfer to hardware it never trained on.

    python scripts/fig/viz_generalisation.py

Writes out/writing/offline-generalisation.png.

The evaluation split answers two different questions and the aggregate answers
neither. 2,400 instances are unseen task sets on platforms the policy trained
on. 4,000 are on the five platforms it never saw. Both are held out, so both are
honest, but only the second is about generalising to hardware.

Only the policy has a training set, so only the policy can be surprised by a
platform. The heuristics are on the chart as the control: they cannot transfer
badly, so whatever they lose between the two groups is the held-out platforms
being harder rather than anyone failing to generalise. The policy's loss is only
interesting relative to theirs.

Grouped bars, not a line. The two pools are disjoint sets of instances, not two
points on a scale, so nothing lies between them and a line joining them would
draw an interpolation that does not exist.

Every allocator has been refined: a constructive pass followed by 4,000
candidate evaluations. No portfolio, since it is six searches rather than one
and belongs to a different cost class.
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
from figstyle import INK, MUTED
from palette import SURFACE

FIGSIZE = (10.4, 4.9)
DPI = 300
PAD = 18
OUTDIR = "writing"

POOLS = ["workload", "hardware"]
POOL_TICK = {"workload": "Unseen workload\non trained platforms",
             "hardware": "Unseen workload\non held-out platforms"}

# each family beside its criticality variant, the policy last
SERIES = ["sa_bfd", "sa_bfd_crit", "sa_wfd", "sa_wfd_crit",
          "sa_heft", "sa_heft_crit", "sa_rl"]


def gather():
    import analyse_allocations as A
    rows = [r for r in A.load() if r["set"] == "eval"]
    out = {}
    for a in SERIES:
        for p in POOLS:
            cell = [r for r in rows
                    if r["allocator"] == a and r.get("pool") == p]
            if cell:
                out[(a, p)] = (sum(r["penalty_hard"] for r in cell) / len(cell),
                               len(cell))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--dir", default=OUTDIR)
    ap.add_argument("--name", default="offline-generalisation")
    args = ap.parse_args()

    import paths
    stats = gather()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE)
    F.chrome(ax)

    width = 0.115
    top = max(v[0] for v in stats.values())
    for i, a in enumerate(SERIES):
        col, _, dash, _, _ = F.style(a)
        # a bar cannot be dashed, so the criticality variants keep the
        # convention by being hatched: same hue, one step darker, a variant
        hatch = "///" if dash != "solid" else None
        xs = [g + (i - (len(SERIES) - 1) / 2) * width
              for g in range(len(POOLS))]
        ys = [stats[(a, p)][0] for p in POOLS]
        ax.bar(xs, ys, width=width * 0.9, color=col, edgecolor=INK, lw=1.0,
               hatch=hatch, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y + top * 0.02, f"{y:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK, zorder=4)

    # the counts belong on the axis rather than in the caption: the two groups
    # are different sizes, and a reader comparing bar heights should be told so
    # where they are looking
    ax.set_xticks(range(len(POOLS)))
    ax.set_xticklabels(
        [f"{POOL_TICK[p]}\n({stats[(SERIES[0], p)][1]:,} task sets)"
         for p in POOLS], fontsize=10.5)
    ax.set_xlim(-0.5, len(POOLS) - 0.5)
    ax.set_ylim(0, top * 1.14)
    ax.set_ylabel("Mean penalty", fontsize=10.5, color=MUTED)
    ax.tick_params(axis="x", length=0)

    handles = [Patch(facecolor=F.style(a)[0], edgecolor=INK, lw=1.0,
                     hatch=("///" if F.style(a)[2] != "solid" else None),
                     label=F.short(a)) for a in SERIES]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(top=0.965, bottom=0.265, left=0.075, right=0.985)
    save(fig, paths.out(args.dir, args.name))
    plt.close(fig)

    n = {p: stats[("sa_rl", p)][1] for p in POOLS}
    print(f"workload {n['workload']:,}   hardware {n['hardware']:,}\n")
    print(f"{'alloc':<13}{'workload':>10}{'hardware':>10}{'ratio':>8}")
    for a in SERIES:
        w, h = stats[(a, "workload")][0], stats[(a, "hardware")][0]
        print(f"{F.short(a):<13}{w:>10.1f}{h:>10.1f}{h / w:>8.2f}")


if __name__ == "__main__":
    main()
