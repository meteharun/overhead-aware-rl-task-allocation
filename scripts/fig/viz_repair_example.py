#!/usr/bin/env python3
"""One worked example of what repair and rebuild actually do.

    python scripts/fig/viz_repair_example.py
    python scripts/fig/viz_repair_example.py --split     three files, to build up a slide

Writes out/slides/repair_example.png, or repair_example_1..3.png with --split.

Three cores, six tasks, one failure. The aggregate charts say repair is cheap
and leaves a penalty while a rebuild is dearer and does not. This says the same
thing on one instance, where you can point at the task that misses.

  1  working    everything fits, nothing is late
  2  repair     C3 dies. Its two tasks are placed in the gaps that already
                exist and nothing else is touched. Two tasks move, and T6 lands
                behind a queue that was not built with it in mind, so it is late
  3  rebuild    every task is placed again from scratch on the two cores that
                are left. Four tasks move and nothing is late

The schedule is hand authored to illustrate, like every other figure of this
kind here. It is not a recorded run. What it is not allowed to do is contradict
itself, so the penalty under each panel is summed from the blocks drawn above
it. The millisecond labels in MS are illustrative on the same scale; the
measured medians live in out/runs/repair.jsonl.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import figstyle as F
import palette as P
import paths
from figstyle import INK, SURFACE

HORIZON = 72.0
KERNEL = 3.0                     # the kernel's cut before any task can start
# thick lanes and a narrow panel. Thin ones over a wide panel turn every task
# into a long strip and the schedule stops reading as blocks of work
LANE_H = 0.78
LANE_Y = {"C1": 2.0, "C2": 1.0, "C3": 0.0}
CORES = ["C1", "C2", "C3"]

# One colour per task, so a task can be followed from panel to panel by colour
# alone. Criticality then has nowhere to live on the block, so it goes in the
# legend label instead, which is where a reader looks for it anyway. None of the
# six is near the red the deadline markers use.
TCOL = {"T1": "#4C78A8", "T2": "#72B7B2", "T3": "#54A24B",
        "T4": "#EECA3B", "T5": "#B279A2", "T6": "#F58518"}
# what a unit of lateness costs, by how much the objective cares
COST = {"HARD": 2.0, "FIRM": 1.2, "SOFT": 0.7}
LANE_FILL = {"C1": P.CPU_RAMP["Kryo_Gen6"][0], "C2": P.CPU_RAMP["Kryo_Gen6"][1],
             "C3": P.UNKNOWN_RAMP[0]}
DEAD_FILL = "#EFE7E7"

# name, criticality, execution, deadline
TASKS = {"T1": ("HARD", 20.0, 40.0), "T2": ("HARD", 14.0, 40.0),
         "T3": ("FIRM", 18.0, 60.0), "T4": ("SOFT", 22.0, 70.0),
         "T5": ("SOFT", 16.0, 70.0), "T6": ("HARD", 10.0, 48.0)}

# who runs where, in order. The three panels are the same six tasks arranged
# three ways, so the eye can follow one task across the figure
WORKING = {"C1": ["T1", "T2"], "C2": ["T3", "T4"], "C3": ["T6", "T5"]}
# C3 is gone. C1 and C2 keep exactly what they had and take the two orphans
REPAIR = {"C1": ["T1", "T2", "T6"], "C2": ["T3", "T4", "T5"], "C3": []}
# nothing is pinned, so T2 and T4 end up on the other core from where they were
REBUILD = {"C1": ["T1", "T6", "T4"], "C2": ["T2", "T3", "T5"], "C3": []}

# illustrative. The measured medians are in out/runs/repair.jsonl, where a
# rebuild includes its 1,000 evaluations of annealing and a repair does not.
MS = {"repair": 3.1, "rebuild": 9.7}

PANELS = [("Working", WORKING, None, None),
          ("Repair", REPAIR, WORKING, MS["repair"]),
          ("Rebuild", REBUILD, WORKING, MS["rebuild"])]

LEGEND = ([(TCOL[n], None, f"{n}   {TASKS[n][0].lower()}") for n in TCOL]
          + [("none", "////", "kernel"), (P.MISSED, None, "late")])


def schedule(plan):
    """Lay each core's list out end to end, kernel then task. Returns blocks and
    the penalty they add up to, so the drawing and the number cannot disagree."""
    blocks, penalty, late_tasks = [], 0.0, []
    for core, names in plan.items():
        t = 0.0
        for name in names:
            crit, dur, dl = TASKS[name]
            blocks.append(dict(core=core, kind="kernel", t=t, d=KERNEL))
            t += KERNEL
            late = max(0.0, t + dur - dl)
            blocks.append(dict(core=core, kind="run", t=t, d=dur, task=name,
                               crit=crit, late=late, deadline=dl))
            if late > 0:
                penalty += late * COST[crit]
                late_tasks.append((name, late))
            t += dur
    return blocks, penalty, late_tasks


def moved(plan, base):
    where = {n: c for c, ns in base.items() for n in ns}
    return sum(1 for c, ns in plan.items() for n in ns if where[n] != c)


def draw_panel(ax, title, plan, base, ms, dead):
    F.chrome(ax)
    ax.grid(False)
    ax.set_xlim(-1.5, HORIZON)
    # headroom over C1 for the deadline flags, nothing below C3
    ax.set_ylim(-0.18, 3.14)
    ax.set_yticks([LANE_Y[c] + LANE_H / 2 for c in CORES])
    ax.set_yticklabels(CORES, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    for core in CORES:
        y = LANE_Y[core]
        gone = dead and core == "C3"
        ax.add_patch(Rectangle((0, y), HORIZON, LANE_H,
                               facecolor=DEAD_FILL if gone else LANE_FILL[core],
                               edgecolor=P.HAIRLINE, lw=1.0, zorder=2))
        if gone:
            ax.plot([0, HORIZON], [y, y + LANE_H], color=F.BAD_X, lw=1.4,
                    zorder=4)
            ax.plot([0, HORIZON], [y + LANE_H, y], color=F.BAD_X, lw=1.4,
                    zorder=4)
            ax.text(HORIZON / 2, y + LANE_H / 2, "core failed", ha="center",
                    va="center", fontsize=10, fontweight="bold", color=F.BAD_X,
                    zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                              edgecolor="none"))

    blocks, penalty, late_tasks = schedule(plan)
    for b in blocks:
        y = LANE_Y[b["core"]]
        if b["kind"] == "kernel":
            ax.add_patch(Rectangle((b["t"], y), b["d"], LANE_H,
                                   facecolor="none", edgecolor=P.GAP, lw=1.1,
                                   hatch="////", zorder=5))
            continue
        fill = TCOL[b["task"]]
        keep = b["d"] - b["late"]
        ax.add_patch(Rectangle((b["t"], y), keep, LANE_H,
                               facecolor=fill, edgecolor=INK, lw=1.0, zorder=5))
        if b["late"] > 0:
            ax.add_patch(Rectangle((b["t"] + keep, y), b["late"], LANE_H,
                                   facecolor=P.MISSED, edgecolor=INK, lw=1.0,
                                   zorder=6))
        ax.text(b["t"] + b["d"] / 2, y + LANE_H / 2, b["task"], ha="center",
                va="center", fontsize=9, fontweight="bold",
                color=P.ink_on(fill), zorder=7)

        # The deadline is a flag above the lane, in the task's own colour, so it
        # is obvious whose deadline it is. A bare tick on the lane was not.
        dl, late = b["deadline"], b["late"]
        ax.plot([dl, dl], [y, y + LANE_H + 0.20],
                color=F.BAD_X if late else fill, lw=1.2,
                linestyle=(0, (2, 2)), zorder=8)
        ax.plot([dl], [y + LANE_H + 0.24], marker="v", markersize=7,
                markerfacecolor=F.BAD_X if late else fill, markeredgecolor=INK,
                markeredgewidth=0.8, zorder=9)

    F.title(ax, title)
    bits = []
    bits.append("nothing late" if penalty == 0 else
                f"{late_tasks[0][0]} is {late_tasks[0][1]:.0f} late, "
                f"penalty {penalty:.0f}")
    if base is not None:
        bits.append(f"{moved(plan, base)} tasks moved")
    if ms is not None:
        bits.append(f"{ms:.0f} ms to decide")
    # clear of the lanes, and wrapped rather than run on. Three facts on one
    # line under a narrow panel is wider than the panel
    ax.set_xlabel("\n".join(bits), fontsize=9.5, labelpad=9,
                  color=F.GOOD if penalty == 0 else INK,
                  fontweight="bold" if penalty else "normal")
    return penalty


def draw(axes, dead_from=1):
    for i, (title, plan, base, ms) in enumerate(PANELS):
        draw_panel(axes[i], title, plan, base, ms, dead=i >= dead_from)


def legend(fig):
    """Down the right hand side, one entry per row. A row along the bottom
    competes with the line of text under each panel."""
    handles = [Patch(facecolor=fc, edgecolor=INK if fc != "none" else P.GAP,
                     hatch=h, lw=1.0, label=lab) for fc, h, lab in LEGEND]
    handles.append(plt.Line2D([], [], color=INK, lw=0, marker="v",
                              markersize=7, markerfacecolor=F.BAD_X,
                              markeredgewidth=0.8, label="deadline"))
    fig.legend(handles=handles, loc="center right", ncol=1, frameon=False,
               fontsize=8.5, labelspacing=0.62, handlelength=1.5,
               bbox_to_anchor=(1.005, 0.52))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="store_true",
                    help="one file per stage, for building the slide up")
    args = ap.parse_args()

    if args.split:
        for i, (title, plan, base, ms) in enumerate(PANELS, start=1):
            fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=170)
            fig.patch.set_facecolor(SURFACE)
            draw_panel(ax, title, plan, base, ms, dead=i >= 2)
            legend(fig)
            fig.subplots_adjust(top=0.86, bottom=0.26, left=0.08, right=0.70)
            out = paths.out("slides", f"repair_example_{i}")
            fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=170,
                             gridspec_kw={"wspace": 0.20})
    fig.patch.set_facecolor(SURFACE)
    draw(axes)
    legend(fig)
    F.suptitle(fig, "One core fails. Patch the hole, or replan from scratch")
    # right stops short of the frame to leave the legend its column, bottom
    # leaves room for the three line caption under each panel
    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.04, right=0.855)
    out = paths.out("slides", "repair_example")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
