#!/usr/bin/env python3
"""How the penalty is calculated, on one worked example.

    python scripts/fig/viz_penalty_example.py

Writes out/writing/penalty-example.png.

Four tasks on two cores, drawn in the same vocabulary as the timing model
figure: blocks laid end to end, hatched kernel work, a send on the sender's
core, a receive on the destination, and the part of a task past its deadline in
black.

Every term of the objective fires once, and each one is attributable to exactly
one thing:

  T4  a hard task that misses its own deadline
  T3  a soft task that misses its own deadline, at a fraction of the cost
  T2  meets its own deadline comfortably, and still costs 150, because the
      chain that runs from T1 across the link and ends at T2 finishes past its
      budget. This is the communication term standing on its own. Nothing on
      the chain missed a deadline
  the class of core each task landed on

T2's own deadline sits to the right of its block and the chain budget sits to
the left of the chain's end, which is the whole distinction the figure exists to
draw.

The schedule is authored, like the other example figures here. What it may not
do is disagree with itself, so the ledger is summed from the same tables the
lanes are drawn from, and the external weights and preference costs come from
`timing` rather than being typed in. Change a block and the arithmetic follows.

No preemption. One task runs to completion before the next starts.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import palette as P
from timing import EXTERNAL_WEIGHT, preference_penalty

FIGSIZE = (12.8, 4.5)
DPI = 300
PAD = 18
OUTDIR = "writing"

INK, MUTED, HAIR = P.INK, P.INK_MUTED, P.HAIRLINE
BAD = "#C62828"
LINK = P.LINK_COLOUR["eth_10g"]
CHAIN = "#5E86A0"

LANE_X0, LANE_X1 = 7.0, 58.0
LANE_H = 11.0
LANE_A_Y, LANE_B_Y = 72.0, 46.0
HORIZON = 78.0
LEDGER_X = 64.0

# T2 and T4 sit on the os core, which is where a receive lands, so the figure
# draws the handler on the core that really carries it. It also gives T2 a
# preference cost of zero, since an os core is where a soft task belongs.
TASKS = {
    "T1": dict(crit="HARD", period=100, weight=1.5, core_class="isolated"),
    "T2": dict(crit="SOFT", period=1000, weight=1.0, core_class="os"),
    "T3": dict(crit="FIRM", period=500, weight=2.0, core_class="isolated"),
    "T4": dict(crit="HARD", period=100, weight=1.0, core_class="os"),
}

LANE_A = [("wake", 0.0, 4.0, ""),
          ("run", 4.0, 16.0, "T1"),
          ("send", 20.0, 5.0, ""),
          ("disp", 25.0, 4.0, ""),
          ("run", 29.0, 37.0, "T3")]
LANE_B = [("wake", 0.0, 4.0, ""),
          ("run", 4.0, 26.0, "T4"),
          ("rx", 30.0, 8.0, ""),
          ("disp", 38.0, 4.0, ""),
          ("run", 42.0, 20.0, "T2")]

DEADLINE = {"T1": 26.0, "T2": 72.0, "T3": 54.0, "T4": 36.0}
FLIGHT = (25.0, 30.0)              # send block to receive block, the link delay
CHAIN_PATH = ("T1", "T2")
CHAIN_END = 62.0                   # T2 completes
CHAIN_BUDGET = 50.0

FILL = {"HARD": P.HARD_FILL, "FIRM": P.FIRM_FILL, "SOFT": P.SOFT_FALLBACK}
KEY = [("HARD", "hard"), ("FIRM", "firm"), ("SOFT", "soft"),
       ("late", "past deadline"), ("budget", "past budget"),
       ("wake", "wake, dispatch"), ("send", "send"), ("rx", "receive")]


def tx(u):
    return LANE_X0 + u / HORIZON * (LANE_X1 - LANE_X0)


def ends(tid):
    for kind, t0, d, lab in LANE_A + LANE_B:
        if kind == "run" and lab == tid:
            return t0 + d
    raise KeyError(tid)


def missed(tid):
    return ends(tid) > DEADLINE[tid]


def cost(tid):
    t = TASKS[tid]
    return t["weight"] * EXTERNAL_WEIGHT[t["period"]]


def pref(tid):
    return preference_penalty(TASKS[tid]["period"], TASKS[tid]["core_class"])


def block(ax, x, y, w, h, kind, z=5):
    if kind == "late":
        ax.add_patch(Rectangle((x, y), w, h, facecolor=P.MISSED, edgecolor=INK,
                               lw=1.0, zorder=z))
    elif kind == "rx":
        ax.add_patch(Rectangle((x, y), w, h, facecolor=LINK, edgecolor=INK,
                               lw=1.0, zorder=z))
    elif kind == "send":
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=LINK,
                               lw=1.5, hatch="\\\\", zorder=z))
    elif kind == "budget":
        # dotted rather than hatched, so it cannot be mistaken for kernel work,
        # and laid over the task block rather than replacing it: this part of
        # T2 still runs, it is just past the chain's budget
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=CHAIN,
                               lw=1.6, hatch="....", zorder=z))
    elif kind in FILL:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=FILL[kind],
                               edgecolor=INK, lw=1.0, zorder=z))
    else:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=P.GAP,
                               lw=1.2, hatch="////", zorder=z))


def lane(ax, y, blocks, fill, name):
    ax.add_patch(Rectangle((LANE_X0, y), LANE_X1 - LANE_X0, LANE_H,
                           facecolor=fill, edgecolor=HAIR, lw=1.1, zorder=2))
    ax.text(LANE_X0 - 1.6, y + LANE_H / 2, name, ha="right", va="center",
            fontsize=10, fontweight="bold", color=INK)

    for kind, t0, d, lab in blocks:
        if kind != "run":
            block(ax, tx(t0), y, tx(t0 + d) - tx(t0), LANE_H, kind)
            continue
        crit, end, d_at = TASKS[lab]["crit"], t0 + d, DEADLINE[lab]
        cut = min(end, d_at)
        block(ax, tx(t0), y, tx(cut) - tx(t0), LANE_H, crit)
        if end > d_at:
            block(ax, tx(cut), y, tx(end) - tx(cut), LANE_H, "late", z=6)
        ax.text(tx((t0 + cut) / 2), y + LANE_H / 2, lab, ha="center",
                va="center", fontsize=10.5, fontweight="bold",
                color=P.ink_on(FILL[crit]), zorder=7)

        late = end > d_at
        ax.plot([tx(d_at)] * 2, [y - 1.8, y + LANE_H + 1.8],
                color=BAD if late else MUTED, lw=1.4,
                linestyle="solid" if late else (0, (3, 2)), zorder=8)
        ax.text(tx(d_at), y + LANE_H + 2.4, f"$D_{{{lab[1:]}}}$", ha="center",
                va="bottom", fontsize=9.5, color=BAD if late else MUTED,
                zorder=8)


def draw(ax):
    lane(ax, LANE_A_Y, LANE_A, P.UNKNOWN_RAMP[1], "isolated")
    lane(ax, LANE_B_Y, LANE_B, P.UNKNOWN_RAMP[0], "os")

    f0, f1 = FLIGHT
    ax.annotate("", xy=(tx(f1), LANE_B_Y + LANE_H), xytext=(tx(f0), LANE_A_Y),
                arrowprops=dict(arrowstyle="-|>", color=LINK, lw=2.0,
                                connectionstyle="arc3,rad=0.3"), zorder=9)
    ax.text(tx((f0 + f1) / 2) + 1.0, (LANE_A_Y + LANE_B_Y + LANE_H) / 2,
            "link delay", ha="left", va="center", fontsize=9, color=LINK)

    # the budget, marked on the lane exactly as a deadline is, because it is the
    # same kind of thing: a line the work has to finish before
    block(ax, tx(CHAIN_BUDGET), LANE_B_Y, tx(CHAIN_END) - tx(CHAIN_BUDGET),
          LANE_H, "budget", z=8)
    ax.plot([tx(CHAIN_BUDGET)] * 2, [LANE_B_Y - 1.8, LANE_B_Y + LANE_H + 1.8],
            color=CHAIN, lw=1.5, linestyle=(0, (4, 3)), zorder=9)
    ax.text(tx(CHAIN_BUDGET), LANE_B_Y + LANE_H + 2.4,
            f"{CHAIN_PATH[0]} to {CHAIN_PATH[1]} budget", ha="center",
            va="bottom", fontsize=9.5, color=CHAIN, fontweight="bold",
            zorder=9)

    ax.text(tx(CHAIN_END / 2), LANE_B_Y - 6.0,
            f"{CHAIN_PATH[0]} and {CHAIN_PATH[1]} both meet their deadlines, "
            f"but the chain from {CHAIN_PATH[0]} to {CHAIN_PATH[1]} "
            f"exceeds its budget",
            ha="center", va="top", fontsize=9.5, color=CHAIN,
            fontweight="bold")

    for i, (kind, label) in enumerate(KEY):
        kx = LANE_X0 + (i % 4) * 13.0
        ky = LANE_B_Y - 14.0 - (i // 4) * 5.6
        block(ax, kx, ky, 3.6, 3.2, kind, z=4)
        ax.text(kx + 4.6, ky + 1.6, label, ha="left", va="center", fontsize=9,
                color=INK)


def ledger(ax):
    late = sorted(t for t in TASKS if missed(t))
    worst = max(CHAIN_PATH, key=cost)
    misses = sum(cost(t) for t in late)
    chain = cost(worst)
    preference = sum(pref(t) for t in TASKS)

    rows = [("head", "Missed deadlines", None)]
    for t in late:
        d = TASKS[t]
        rows.append(("item", f"  {t}  {d['crit']:<4} w {d['weight']:.1f} x e "
                             f"{EXTERNAL_WEIGHT[d['period']]:<3.0f}",
                     f"{cost(t):>7.2f}"))
    rows.append(("sum", "  " + " + ".join(f"{cost(t):.2f}" for t in late),
                 f"{misses:>7.2f}"))

    # priced exactly like a miss, and shown that way, because "150" with no
    # working is the line a reader stops at
    rows.append(("head", "Chain budget overrun", None))
    rows.append(("item", "  priced at the highest miss cost on the path", None))
    rows.append(("item", f"  {worst}  {TASKS[worst]['crit']:<4} w "
                         f"{TASKS[worst]['weight']:.1f} x e "
                         f"{EXTERNAL_WEIGHT[TASKS[worst]['period']]:<3.0f}",
                 f"{chain:>7.2f}"))

    # only the placements that cost something. A row of zeros is a row of noise
    rows.append(("head", "Core preference", None))
    for t in sorted(t for t in TASKS if pref(t) > 0):
        rows.append(("item",
                     f"  {t}  {TASKS[t]['crit']:<4} on {TASKS[t]['core_class']}",
                     f"{pref(t):>7.2f}"))
    rows.append(("sum", "", f"{preference:>7.2f}"))

    y, step = 94.0, 5.0
    for kind, text, value in rows:
        if kind == "head":
            y -= step * 0.35
            ax.text(LEDGER_X, y, text, ha="left", va="center", fontsize=10,
                    fontweight="bold", color=INK)
        elif kind == "sum":
            ax.plot([LEDGER_X + 20.0, 100.0], [y + 2.9, y + 2.9], color=HAIR,
                    lw=0.9, zorder=3)
            if text:
                ax.text(LEDGER_X, y, text, ha="left", va="center", fontsize=8.5,
                        color=MUTED, family="monospace")
            ax.text(100.0, y, value, ha="right", va="center", fontsize=9,
                    color=INK, family="monospace")
        else:
            ax.text(LEDGER_X, y, text, ha="left", va="center", fontsize=9,
                    color=INK, family="monospace")
            if value:
                ax.text(100.0, y, value, ha="right", va="center", fontsize=9,
                        color=INK, family="monospace")
        y -= step

    # one total, with the three terms spelled out beside it. `penalty_hard` is
    # the code's name for misses plus chain and it reads as though it were about
    # hard tasks, which it is not, so the figure does not use it at all
    y -= step * 0.4
    total = misses + chain + preference
    ax.text(LEDGER_X, y, "Total penalty", ha="left", va="center", fontsize=11,
            fontweight="bold", color=INK)
    ax.text(100.0, y, f"{total:>7.2f}", ha="right", va="center", fontsize=11,
            fontweight="bold", color=INK, family="monospace")
    ax.text(LEDGER_X, y - step * 0.85,
            f"{misses:.2f} + {chain:.2f} + {preference:.2f}", ha="left",
            va="center", fontsize=9, color=MUTED, family="monospace")
    return misses, chain, preference


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
    ap.add_argument("--name", default="penalty-example")
    args = ap.parse_args()

    import paths
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    fig.patch.set_facecolor(P.SURFACE)
    ax.set_xlim(0, 101)
    ax.set_ylim(12, 100)
    ax.axis("off")

    draw(ax)
    misses, chain, preference = ledger(ax)

    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.005, right=0.995)
    save(fig, paths.out(args.dir, args.name))
    plt.close(fig)

    for t in sorted(TASKS):
        print(f"{t}  {TASKS[t]['crit']:<5}on {TASKS[t]['core_class']:<9}"
              f"ends {ends(t):>5.0f}  D {DEADLINE[t]:>5.0f}  "
              f"{'MISS' if missed(t) else 'ok':<5}m={cost(t):>6.1f}  "
              f"pref={pref(t):.2f}")
    print(f"chain {CHAIN_PATH[0]}->{CHAIN_PATH[1]}  {CHAIN_END:.0f} > "
          f"{CHAIN_BUDGET:.0f}, charged {chain:.2f}")
    print(f"misses {misses:.2f}  chain {chain:.2f}  preference {preference:.2f}"
          f"  hard {misses + chain:.2f}  total "
          f"{misses + chain + preference:.2f}")


if __name__ == "__main__":
    main()
