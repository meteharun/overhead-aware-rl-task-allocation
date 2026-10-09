#!/usr/bin/env python3
"""Timing model inputs and output.

    python scripts/fig/viz_model_io.py

Writes out/writing/model_io.png.

Two cores on the output side, on two devices, so a crossing has somewhere to
happen. The sender pays a send on its own core, the message spends the link
delay in flight, and the receive lands on the destination device's os core
above the application task already running there.

Every word in the figure is a label. The explanation belongs in the body text.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import palette as P
import paths

FIGSIZE = (11.0, 4.6)
DPI = 300
PAD = 18
OUTDIR = "writing"

INK, MUTED, HAIR = P.INK, P.INK_MUTED, P.HAIRLINE
BAD = "#C62828"
LINK = P.LINK_COLOUR["eth_10g"]
MODEL_FILL = "#EAF0F8"
MODEL_EDGE = "#1B4F9C"

IN_X, IN_W, IN_H = 3.0, 17.0, 13.0
INPUTS = [("Task set", 62.0), ("Platform", 46.0), ("Allocation", 30.0)]
CAT_BOX = (3.0, 12.0, 17.0, 10.0)
MODEL_BOX = (28.0, 30.0, 17.0, 45.0)

LANE_X0, LANE_X1 = 61.5, 97.0
LANE_H = 9.0
LANE_A_Y, LANE_B_Y = 70.0, 48.0
HORIZON = 100.0

# A1 holds two tasks. T_i has the shorter deadline so it preempts T_j, and it
# is T_i that produces data for T_k on the other device. B1 is that device's os
# core, so the receive lands there at top priority and interrupts T_k.
#   kind      start  dur   label
LANE_A = [("wake",  0.0,  4.0, ""),
          ("run_j", 4.0, 10.0, "$T_j$"),
          ("disp", 14.0,  4.0, ""),
          ("run_i", 18.0, 12.0, "$T_i$"),
          ("send", 30.0,  4.0, ""),
          ("pre",  34.0,  6.0, ""),
          ("run_j", 40.0, 10.0, "$T_j$"),
          ("tail", 50.0,  5.0, "")]
LANE_B = [("wake",  0.0,  4.0, ""),
          ("run_k", 4.0, 37.0, "$T_k$"),
          ("disp", 41.0,  4.0, ""),
          ("rx",   45.0,  8.0, ""),
          ("pre",  53.0,  6.0, ""),
          ("run_k", 59.0, 11.0, "$T_k$"),
          ("tail", 70.0,  6.0, "")]
FLIGHT = (34.0, 41.0)               # link delay, sender's core to the os core
R_AT, D_AT = 76.0, 86.0

KEY = [("wake", "wake"), ("disp", "dispatch"), ("send", "send"),
       ("pre", "resume"), ("tail", "tail"), ("rx", "receive")]
KEY_COLS = 3


def tx(u):
    return LANE_X0 + u / HORIZON * (LANE_X1 - LANE_X0)


def rbox(ax, x, y, w, h, fc, ec, lw=1.4, z=3, r=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=z))


FILL = {"run_i": P.HARD_FILL, "run_j": P.FIRM_FILL,
        "run_k": P.SOFT_FALLBACK}


def block(ax, x, y, w, h, kind, z=5):
    if kind in FILL:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=FILL[kind],
                               edgecolor=INK, lw=1.0, zorder=z))
    elif kind == "rx":
        ax.add_patch(Rectangle((x, y), w, h, facecolor=LINK, edgecolor=INK,
                               lw=1.0, zorder=z))
    elif kind == "send":
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=LINK,
                               lw=1.5, hatch="\\\\", zorder=z))
    elif kind == "tail":
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=INK,
                               lw=1.0, ls=(0, (3, 2)), zorder=z))
    elif kind == "pre":
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=P.GAP,
                               lw=1.2, hatch="xx", zorder=z))
    else:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=P.GAP,
                               lw=1.2, hatch="////", zorder=z))


def lane(ax, y, blocks, fill, name):
    ax.add_patch(Rectangle((LANE_X0, y), LANE_X1 - LANE_X0, LANE_H,
                           facecolor=fill, edgecolor=HAIR, lw=1.1, zorder=2))
    ax.text(LANE_X0 - 1.4, y + LANE_H / 2, name, ha="right", va="center",
            fontsize=10, fontweight="bold", color=INK)
    for kind, t0, d, lab in blocks:
        block(ax, tx(t0), y, tx(t0 + d) - tx(t0), LANE_H, kind)
        if lab:
            ax.text(tx(t0 + d / 2), y + LANE_H / 2, lab, ha="center",
                    va="center", fontsize=11, color=P.ink_on(FILL[kind]),
                    zorder=7)


def draw(ax):
    mx, my, mw, mh = MODEL_BOX
    for name, y in INPUTS:
        rbox(ax, IN_X, y, IN_W, IN_H, P.DEVICE_FILL, HAIR, lw=1.5)
        ax.text(IN_X + IN_W / 2, y + IN_H / 2, name, ha="center", va="center",
                fontsize=12, color=INK)
        ax.annotate("", xy=(mx, y + IN_H / 2), xytext=(IN_X + IN_W, y + IN_H / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5),
                    zorder=4)

    cx, cy, cw, ch = CAT_BOX
    rbox(ax, cx, cy, cw, ch, "white", HAIR, lw=1.3, r=0.7)
    ax.text(cx + cw / 2, cy + ch / 2, "Catalogues", ha="center", va="center",
            fontsize=11, color=MUTED)
    ax.annotate("", xy=(mx + mw / 2, my), xytext=(cx + cw, cy + ch / 2),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                connectionstyle="angle,angleA=0,angleB=90,rad=4"),
                zorder=4)

    rbox(ax, mx, my, mw, mh, MODEL_FILL, MODEL_EDGE, lw=2.0, r=1.2)
    ax.text(mx + mw / 2, my + mh / 2, "Timing\nmodel", ha="center", va="center",
            fontsize=14, fontweight="bold", color=MODEL_EDGE, linespacing=1.3)
    mid = (LANE_A_Y + LANE_B_Y + LANE_H) / 2
    ax.annotate("", xy=(LANE_X0 - 1.5, mid), xytext=(mx + mw, mid),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5), zorder=4)

    # both lanes stay neutral grey so the pink link marks are the only strong
    # colour on the right. The A55 red tint fought the send and receive blocks
    lane(ax, LANE_A_Y, LANE_A, P.UNKNOWN_RAMP[1], "A1")
    lane(ax, LANE_B_Y, LANE_B, P.UNKNOWN_RAMP[0], "B1")

    # the message in flight, from the end of the send to where it lands
    f0, f1 = FLIGHT
    ax.annotate("", xy=(tx(f1), LANE_B_Y + LANE_H), xytext=(tx(f0), LANE_A_Y),
                arrowprops=dict(arrowstyle="-|>", color=LINK, lw=2.0),
                zorder=6)
    ax.text(tx((f0 + f1) / 2) + 1.0, (LANE_A_Y + LANE_B_Y + LANE_H) / 2,
            "delay", ha="left", va="center", fontsize=9.5, color=LINK)

    y0 = LANE_B_Y - 4.0
    ax.annotate("", xy=(tx(R_AT), y0), xytext=(tx(0), y0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2), zorder=6)
    ax.text(tx(R_AT / 2), y0 - 1.4, "$R_k$", ha="center", va="top",
            fontsize=11, color=INK)
    ax.plot([tx(D_AT)] * 2, [LANE_B_Y - 1.6, LANE_B_Y + LANE_H + 1.6],
            color=BAD, lw=1.8, zorder=8)
    ax.text(tx(D_AT), LANE_B_Y + LANE_H + 2.2, "$D_k$", ha="center",
            va="bottom", fontsize=11, color=BAD)

    col = (LANE_X1 - LANE_X0) / KEY_COLS
    for i, (kind, label) in enumerate(KEY):
        kx = LANE_X0 + (i % KEY_COLS) * col
        ky = LANE_B_Y - 13.0 - (i // KEY_COLS) * 5.0
        block(ax, kx, ky, 2.8, 2.6, kind, z=6)
        ax.text(kx + 3.6, ky + 1.3, label, ha="left", va="center",
                fontsize=9.5, color=INK)


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

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    draw(ax)
    save(fig, paths.out(args.dir, "model_io"))
    plt.close(fig)


if __name__ == "__main__":
    main()
