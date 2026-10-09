#!/usr/bin/env python3
"""Worked examples of one task set and one platform, for the thesis.

    python scripts/fig/gen_example_figs.py
    python scripts/fig/gen_example_figs.py --durations
    python scripts/fig/gen_example_figs.py --dpi 200 --no-legend

Writes out/writing/example_taskset.png and out/writing/example_platform.png.
--dir slides puts them in out/slides instead, for a presentation.

viz_taskset and viz_platform draw the real thing, which at forty tasks and
sixteen cores does not survive a projector. These two are the illustration
instead, on the same palette and primitives as the videos.

The task set is named after the STATCOM workload in
reference/STATCOM_IMX95_REAL_TASK_WORKLOAD.md rather than T1 to T6, so the
application is visible. It is the measurement to protection to control to
actuator chain from section 6 of that document, plus the recorder the
protection check trips and a supervisory state machine on its own clock: six of
the real fourteen. --generic falls back to T1 to T6.

Execution times are off the labels by default; --durations puts them back.

figparts is imported for its palette and drawing helpers and is not modified.
Both panels are drawn here rather than by its functions, since the task names
are wider than the bars and have to sit underneath, and device B has three
cores where draw_agent_right has fixed boxes and a fixed time axis.

The two images are cropped to their own content, so they can be placed side by
side and scaled freely.
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
import palette as P
import paths

FIGSIZE = (10, 5.8)        # same as the video, so the geometry is the video's
PAD = 18                   # px of white kept around the content, at DPI 300
DPI = 300
OUTDIR = "writing"         # out/writing. Thesis figures, not slide figures

LEG_X, LEG_Y = 5.0, 33.5   # criticality key, just under the last task
HEAD_X, HEAD_Y = 46.0, 95.5

# ------------------------- the STATCOM task set ---------------------------
# Six of the real fourteen tasks from
# reference/STATCOM_IMX95_REAL_TASK_WORKLOAD.md: the protection and control
# chain section 6 describes, plus the recorder the protection check trips and a
# supervisory state machine on its own clock.
#
# Bar width is execution time and colour is criticality, as everywhere else.
# Four of the six are hard because on a real control platform most of the graph
# is hard.
#
# Execution times are relative rather than measured, but the ordering is not
# arbitrary: a task on a 100 us period cannot do much work per release, so the
# four hard ones are the short bars, and the recorder runs rarely and does a
# great deal when it does, which is why it is dangerous to place badly.
#
# The four chain tasks sit at 100 us because each inherits its producer's
# period, which shows the rule as four equal periods rather than as an edge
# annotation. Deadlines are the model's 0.8 to 1.0 of the period, shown as D/T.
#
#          key          label                       crit    exec   T     D
TS_TASKS = [("ingest",  "T1 - measurement\ningest",  "HARD", 10.0,  100,   92),
            ("protect", "T2 - protection\ncheck",    "HARD",  7.0,  100,   85),
            ("control", "T3 - control loop",         "HARD", 14.0,  100,   96),
            ("actuate", "T4 - actuator\ncommand",    "HARD",  6.0,  100,   88),
            ("superv",  "T5 - supervisory\nstate",   "FIRM", 18.0,  500,  430),
            ("record",  "T6 - fault recorder",       "SOFT", 34.0, None, None)]
TS_EDGES = [("ingest", "protect"), ("protect", "control"),
            ("control", "actuate"), ("protect", "record")]
# spaced so the labels underneath clear each other: `T1 - measurement` is about
# thirteen axis units wide, so the gaps are fourteen
TS_NODE = {"ingest": (8.0, 64.0), "protect": (27.5, 64.0),
           "control": (45.35, 64.0), "actuate": (67.05, 64.0),
           "superv": (8.0, 38.0), "record": (40.0, 38.0)}
TS_SCALE = 0.55            # axis units per unit of execution time
TS_NODE_H = 5.0
TS_LEG_Y = 19.0
CRIT = {"HARD": P.HARD_FILL, "FIRM": P.FIRM_FILL, "SOFT": P.SOFT_FALLBACK}

# The platform panel. Devices side by side, cores stacked inside them, and the
# link running horizontally between the two boxes. No time axis: nothing is
# scheduled here, this is the hardware the allocator is handed.
#
# Each core states its CPU type and core class rather than relying on colour
# alone, since the chapter this figure belongs to has not yet defined the
# palette. The fill still follows the project encoding, hue for CPU type and
# lightness for core class.
#
# CPU types and the link are named generically, because this chapter defines
# the platform model rather than the catalogue. What matters is only whether
# two cores share a type.
#
# The assignment shows three things at once: device A is internally
# heterogeneous, device B is uniform, and A1 and B1 are the same type on
# different devices, so CPU type and device read as independent axes.
#
#              label       x     y     w     h     cores
PL_DEVICES = [("Device A",  6.0, 30.0, 38.0, 40.0, ["A1", "A2"]),
              ("Device B", 56.0, 22.0, 38.0, 56.0, ["B1", "B2", "B3"])]
# display name, palette key, core class
PL_CORE = {"A1": ("CPU type 1", "A55", "os"),
           "A2": ("CPU type 2", "A78", "isolated"),
           "B1": ("CPU type 1", "A55", "os"),
           "B2": ("CPU type 3", "Kryo_Gen6", "general"),
           "B3": ("CPU type 3", "Kryo_Gen6", "isolated")}
PL_CORE_Y = {"A1": 50.0, "A2": 34.0, "B1": 58.0, "B2": 42.0, "B3": 26.0}
PL_CORE_H = 12.0
PL_CORE_PAD = 3.0                  # inset of a core box inside its device box
PL_LINK_Y = 50.0                   # the vertical centre of both device boxes
PL_LINK_LABEL = "link"
PL_LINK_COL = P.LINK_COLOUR["eth_10g"]


def _extent(ax, artist):
    """A drawn label's box, in axis units. Measured, not guessed."""
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    bb = artist.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x0, y0, x1, y1


def text_width(ax, artist):
    x0, _, x1, _ = _extent(ax, artist)
    return x1 - x0


def text_bottom(ax, artist):
    """Where a label actually ends. A two line name at ten point is over six
    axis units tall, and arithmetic on the font size gets that wrong."""
    return _extent(ax, artist)[1]


def crit_legend(ax, x0, y):
    """HARD / FIRM / SOFT, using the same swatches the video's legend uses.

    Laid out by measuring each word, since they differ in length and a fixed
    pitch either collides or leaves a hole.
    """
    sw, h = 3.4, 3.0
    x = x0
    for key in ("HARD", "FIRM", "SOFT"):
        gv.ag_swatch(ax, x, y, sw, h, "crit", key)
        label = ax.text(x + sw + 1.3, y + h / 2, key, ha="left", va="center",
                        fontsize=10.5, color=gv.INK)
        x += sw + 1.3 + text_width(ax, label) + 5.0


def strip_durations(ax):
    """`T4 · 30` becomes `T4`, on the labels the left panel just drew."""
    short = {f"{n} · {d}": n for n, _, d in gv.AG_TASKS}
    hit = 0
    for t in ax.texts:
        if t.get_text() in short:
            t.set_text(short[t.get_text()])
            hit += 1
    if hit != len(short):
        print(f"  warning: stripped {hit} of {len(short)} task labels. "
              f"figparts may have changed the label format")


def save(fig, path, pad=PAD):
    """Render, crop to whatever was actually drawn, write it out."""
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    flat = Image.new("RGB", im.size, im.getpixel((0, 0)))
    bbox = (ImageChops.difference(im, flat).convert("L")
            .point(lambda v: 255 if v > 6 else 0).getbbox())
    if bbox is None:
        raise SystemExit(f"nothing was drawn for {path.name}")
    x0, y0, x1, y1 = bbox
    im = im.crop((max(0, x0 - pad), max(0, y0 - pad),
                  min(im.size[0], x1 + pad), min(im.size[1], y1 + pad)))
    im.save(path)
    print(f"wrote {path}  ({im.size[0]}x{im.size[1]})")


def blank(dpi):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def draw_taskset(ax, durations=False):
    """The STATCOM task set: bars sized by execution time, named underneath.

    The name goes under the bar rather than beside it. Beside it works for T1
    and T4, and does not work at all for `measurement ingest`, which is wider
    than the bar it belongs to.
    """
    ax.text(8.0, 73.0, "Task set", fontsize=12, fontweight="bold", color=gv.MUTED)
    span = {}
    for key, label, crit, dur, period, deadline in TS_TASKS:
        x, y = TS_NODE[key]
        w = dur * TS_SCALE
        span[key] = (x, y, w)
        gv.box(ax, x, y, w, TS_NODE_H, CRIT[crit], ec=gv.INK, lw=1.2, z=4)
        name = ax.text(x + w / 2, y - 1.6, label, ha="center", va="top",
                       fontsize=10, color=gv.INK, linespacing=1.35)
        # deadline over period, one step down in weight so the name reads
        # first, placed under wherever the name really ended
        # an event triggered task has neither
        sub = "event-triggered" if period is None else f"{deadline} / {period} µs"
        if durations:
            sub += f"  ·  C {dur:.0f}"
        ax.text(x + w / 2, text_bottom(ax, name) - 1.0, sub,
                ha="center", va="top", fontsize=8.5, color=gv.MUTED)

    for a, b in TS_EDGES:
        xa, ya, wa = span[a]
        xb, yb, wb = span[b]
        if abs(ya - yb) < 1e-9:
            # along the chain, out of one bar's right edge into the next
            p0 = (xa + wa, ya + TS_NODE_H / 2)
            p1 = (xb, yb + TS_NODE_H / 2)
            rad = 0.0
        else:
            # the branch off the chain, out of the bottom into the top
            p0 = (xa + wa / 2, ya)
            p1 = (xb + wb / 2, yb + TS_NODE_H)
            rad = -0.16
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="->", color=P.HAIRLINE, lw=1.8,
                                    connectionstyle=f"arc3,rad={rad}"), zorder=3)


def taskset(dpi, durations, legend, generic=False, outdir=OUTDIR):
    fig, ax = blank(dpi)
    if generic:
        # the video's own six, for when the figure has to match rl.mp4
        gv.draw_agent_left(ax, None, None, False, placed=(), placing=None, t=1.0)
        if not durations:
            strip_durations(ax)
        if legend:
            crit_legend(ax, LEG_X, LEG_Y)
    else:
        draw_taskset(ax, durations)
        if legend:
            crit_legend(ax, TS_NODE["superv"][0], TS_LEG_Y)
    save(fig, paths.out(outdir, "example_taskset"))
    plt.close(fig)


def draw_platform(ax):
    """Two devices, five cores, one link. The hardware, with nothing on it."""
    for label, x, y, w, h, cores in PL_DEVICES:
        gv.box(ax, x, y, w, h, P.DEVICE_FILL, ec=P.HAIRLINE, lw=1.6, z=1)
        ax.text(x + PL_CORE_PAD, y + h - 5.0, label, fontsize=12,
                fontweight="bold", color=gv.MUTED)
        for c in cores:
            cpu, key, klass = PL_CORE[c]
            fill = P.core_fill(key, klass)
            cx, cw = x + PL_CORE_PAD, w - 2 * PL_CORE_PAD
            cy = PL_CORE_Y[c]
            gv.box(ax, cx, cy, cw, PL_CORE_H, fill, ec=P.HAIRLINE, lw=1.2, z=2)
            ink = P.ink_on(fill)
            ax.text(cx + 2.2, cy + PL_CORE_H / 2, c, ha="left", va="center",
                    fontsize=13, fontweight="bold", color=ink)
            ax.text(cx + cw - 2.2, cy + PL_CORE_H / 2,
                    "\n".join((cpu, klass)), ha="right", va="center",
                    fontsize=10, color=ink, linespacing=1.3)

    # the link, box edge to box edge, in the clear band between the devices
    a = PL_DEVICES[0]
    b = PL_DEVICES[1]
    x0, x1 = a[1] + a[3], b[1]
    ax.annotate("", xy=(x1, PL_LINK_Y), xytext=(x0, PL_LINK_Y),
                arrowprops=dict(arrowstyle="-", color=PL_LINK_COL, lw=2.6),
                zorder=3)
    ax.text((x0 + x1) / 2, PL_LINK_Y + 1.8, PL_LINK_LABEL, ha="center",
            va="bottom", fontsize=10, color=PL_LINK_COL, fontweight="bold")


def platform(dpi, heading, outdir=OUTDIR):
    fig, ax = blank(dpi)
    draw_platform(ax)
    if heading:
        # the video has no title over this half. Side by side with the task set,
        # which does, one bare panel looks unfinished
        ax.text(PL_DEVICES[0][1], 84.0, "Platform", fontsize=12,
                fontweight="bold", color=gv.MUTED)
    save(fig, paths.out(outdir, "example_platform"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=int, default=DPI,
                    help="output resolution. Layout does not change with it")
    ap.add_argument("--durations", action="store_true",
                    help="keep the execution time on each task label")
    ap.add_argument("--no-legend", dest="legend", action="store_false",
                    help="drop the HARD / FIRM / SOFT key from the task set")
    ap.add_argument("--no-heading", dest="heading", action="store_false",
                    help="drop the word Platform from the platform figure")
    ap.add_argument("--generic", action="store_true",
                    help="the video's T1 to T6 instead of the STATCOM names")
    ap.add_argument("--dir", default=OUTDIR,
                    help="subdirectory of out/. writing for the thesis, "
                         "slides for a presentation")
    args = ap.parse_args()

    taskset(args.dpi, args.durations, args.legend, args.generic, args.dir)
    platform(args.dpi, args.heading, args.dir)


if __name__ == "__main__":
    main()
