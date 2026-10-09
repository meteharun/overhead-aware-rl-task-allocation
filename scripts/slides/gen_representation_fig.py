#!/usr/bin/env python3
"""Render the two TikZ figures in out/slides as PNGs.

    python scripts/slides/gen_representation_fig.py

Writes out/slides/learned_representation.png      (thesis Figure 5.2)
       out/slides/learned_policy_architecture.png (thesis Figure 5.3)

Both sources are bare tikzpictures and compiling them needs a LaTeX
installation that is not on this machine. These redraw the same diagrams with
matplotlib, following the coordinates in the tikz source so the proportions
match. If a LaTeX toolchain is available, compiling the tex is the better
option and this file can be ignored.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

from palette import INK, INK_MUTED, SURFACE

SLIDES = ROOT / "out" / "slides"

TASK_EDGE, TASK_FILL = "#2E4C7E", "#E8EEF7"
DEV_A_EDGE, DEV_A_FILL = "#B5762A", "#FBF3E8"
DEV_B_EDGE, DEV_B_FILL = "#4A7A46", "#EDF4EC"
GREY_EDGE, GREY_FILL = "#8A8A82", "#F2F2EF"


def box(ax, x, y, w, h, text, fill, edge, size=10.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.3,
                                zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK,
            zorder=4, linespacing=1.4)


def arrow(ax, a, b, shrink=2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                 color=INK, linewidth=1.5,
                                 shrinkA=shrink, shrinkB=shrink, zorder=2))


# ---------------------------------------------------------------- Figure 5.2

def representation(out, dpi):
    fig, ax = plt.subplots(figsize=(11.6, 4.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.text(1.8, 1.72, "Task graph", ha="center", fontsize=13,
            fontweight="bold", color=INK)
    pos = {"t1": (0.0, 0.0), "t2": (1.8, 0.7), "t3": (1.8, -0.7),
           "t4": (3.6, 0.0)}
    for a, b in (("t1", "t2"), ("t1", "t3"), ("t2", "t4"), ("t3", "t4")):
        arrow(ax, pos[a], pos[b], shrink=29)
    for name, label in (("t1", r"$\tau_1$"), ("t2", r"$\tau_2$"),
                        ("t3", r"$\tau_3$"), ("t4", r"$\tau_4$")):
        x, y = pos[name]
        ax.add_patch(Circle((x, y), 0.40, facecolor=TASK_FILL,
                            edgecolor=TASK_EDGE, linewidth=1.5, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=12, color=INK,
                zorder=4)
    ax.text(1.8, -1.62, "13 features per task\nDirectional aggregation",
            ha="center", va="top", fontsize=11, color=INK_MUTED)

    ax.text(8.1, 1.72, "Hardware and links", ha="center", fontsize=13,
            fontweight="bold", color=INK)
    for x0, x1, fill, edge, name, cx in (
            (5.0, 7.2, DEV_A_FILL, DEV_A_EDGE, "Device A", 6.1),
            (9.0, 11.2, DEV_B_FILL, DEV_B_EDGE, "Device B", 10.1)):
        ax.add_patch(FancyBboxPatch((x0, -1.10), x1 - x0, 2.20,
                                    boxstyle="round,pad=0.02,rounding_size=0.09",
                                    facecolor=fill, edgecolor=edge,
                                    linewidth=1.4, zorder=1))
        ax.text(cx, 0.80, name, ha="center", va="center", fontsize=11,
                color=INK, zorder=4)
    for cx, cy, label in ((6.1, 0.25, "$q_1$: OS"), (6.1, -0.50, "$q_2$: general"),
                          (10.1, 0.25, "$q_3$: OS"),
                          (10.1, -0.50, "$q_4$: isolated")):
        box(ax, cx, cy, 1.56, 0.34, label, "white", "#888880", size=10.5)

    ax.plot([7.2, 9.0], [0.0, 0.0], color=INK, linewidth=1.8, zorder=2)
    ax.text(8.1, 0.14, r"$\ell$", ha="center", va="bottom", fontsize=12,
            color=INK)
    ax.text(8.1, -0.16, r"$h_\ell,\ \delta_\ell$", ha="center", va="top",
            fontsize=11, color=INK)
    ax.text(8.1, -1.62,
            "10 static features per core\nRoutes contribute to task-core features",
            ha="center", va="top", fontsize=11, color=INK_MUTED)

    ax.set_xlim(-0.75, 11.95)
    ax.set_ylim(-2.45, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- Figure 5.3

def policy(out, dpi):
    fig, ax = plt.subplots(figsize=(11.0, 4.6), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    W, H = 3.2, 0.95
    nodes = {
        "task":     (0.0, 2.0, W, "Task features: 13\n" + r"Linear $13\to64$",
                     TASK_FILL, TASK_EDGE),
        "graph":    (0.0, 0.7, W, "3 graph layers\nSelected task: 64",
                     TASK_FILL, TASK_EDGE),
        "hardware": (0.0, -0.8, W, "Core features: 10\n" + r"MLP $10\to64\to64$",
                     DEV_B_FILL, DEV_B_EDGE),
        "pair":     (0.0, -2.2, W, "Pair features: 18\nDynamic inputs",
                     DEV_A_FILL, DEV_A_EDGE),
        "concat":   (4.2, 0.0, 2.4, "Concatenate\n" + r"$64+64+18$" + "\n146 features",
                     GREY_FILL, GREY_EDGE),
        "head":     (7.8, 0.0, W, "Shared scoring MLP\n"
                     + r"$146\to128\to128\to1$", GREY_FILL, GREY_EDGE),
        "action":   (7.8, -1.8, W, "Core scores: $M$\nSample / argmax",
                     GREY_FILL, GREY_EDGE),
    }
    for x, y, w, text, fill, edge in nodes.values():
        box(ax, x, y, w, H, text, fill, edge)

    def edge_pt(name, where):
        x, y, w, *_ = nodes[name]
        dx = {"e": w / 2, "w": -w / 2, "c": 0.0}[where[0]]
        dy = {"n": H / 2, "s": -H / 2, "c": 0.0}[where[1]]
        return (x + dx, y + dy)

    arrow(ax, edge_pt("task", "cs"), edge_pt("graph", "cn"))
    arrow(ax, edge_pt("graph", "ec"), edge_pt("concat", "wn"))
    arrow(ax, edge_pt("hardware", "ec"), edge_pt("concat", "wc"))
    arrow(ax, edge_pt("pair", "ec"), edge_pt("concat", "ws"))
    arrow(ax, edge_pt("concat", "ec"), edge_pt("head", "wc"))
    arrow(ax, edge_pt("head", "cs"), edge_pt("action", "cn"))

    ax.set_xlim(-2.1, 9.8)
    ax.set_ylim(-2.95, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()
    SLIDES.mkdir(parents=True, exist_ok=True)
    representation(SLIDES / "learned_representation.png", args.dpi)
    policy(SLIDES / "learned_policy_architecture.png", args.dpi)
