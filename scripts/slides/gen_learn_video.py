#!/usr/bin/env python3
"""How the agent learns -> learning.mp4. A standalone video, about twenty six seconds.

    python scripts/slides/gen_learn_video.py
    python scripts/slides/gen_learn_video.py --series      print the penalties, draw nothing
    python scripts/slides/gen_learn_video.py --speed 0.6   slower still
    python scripts/slides/gen_learn_video.py --gif

Needs ffmpeg. If it is not on PATH:  pip install imageio-ffmpeg
(or set the FFMPEG environment variable to the ffmpeg.exe you have).

rl.mp4 shows the agent deciding. This one shows it being trained, and it says
one thing only: it makes an allocation, it gets one number back, and it does
that a great many times.

Two regions on screen. Four core lanes at the top holding the current
allocation, and a strip at the bottom holding every penalty so far. Nothing
else. There is no baseline panel, no policy internals and no credit assignment,
because none of that is needed to show the loop, and all of it made the picture
unreadable.

There is also no narration on screen. Captions that change with the beat go past
faster than anyone reads them, so the words belong under the video where the
audience can take them in at once.

Every attempt is a different task set, which is what training does: thirty two
fresh ones per optimiser step. That also keeps the video honest. A single task
set improving over three attempts would claim that one update fixes an
allocation, which is not true and not what the method claims.

The allocations are generated, not recorded. The penalty under each one is
computed from the blocks that are drawn, so the number on screen always matches
the picture, but the falling curve is an illustration of training rather than a
log of it. Run with --series to see the numbers without rendering anything.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
# figparts is the shared drawing library and stays in scripts/fig
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fig"))

import figparts as gv
import palette as P

INK, MUTED, GOOD, BAD, ACCENT = gv.INK, gv.MUTED, gv.GOOD, gv.BAD, gv.ACCENT

FIGSIZE = (10, 5.8)
DPI = 170
FPS = 30
PAD = 6

# ----------------------------- the picture --------------------------------
GX, GW = 12.0, 62.0            # the lanes
HORIZON = 100.0
TS = GW / HORIZON
LANE_H = 5.5
LANES = {"A1": 81.5, "A2": 75.0, "B1": 65.5, "B2": 59.0}
RULE_Y = 72.8                  # the device boundary, between A2 and B1
LANE_FILL = {"A1": P.CPU_RAMP["Kryo_Gen6"][0], "A2": P.CPU_RAMP["Kryo_Gen6"][1],
             "B1": P.UNKNOWN_RAMP[0], "B2": P.UNKNOWN_RAMP[1]}
LINK_COL = P.LINK_COLOUR["eth_10g"]

PEN_X = 86.0                   # the one number, beside the lanes
PEN_LABEL_Y, PEN_Y = 84.5, 74.0

STRIP_X0, STRIP_X1 = 12.0, 97.0
STRIP_BASE, STRIP_H = 26.0, 22.0
STRIP_TITLE_Y = 51.0
COUNT_Y = 21.5
LEG_Y = 14.5

# Playback speed. The captions used to carry the narration and went past too
# fast to read, so they are gone and the words live under the video instead.
# Nothing on screen has to be read now, but the eye still needs time on the
# strip, so everything runs long
SPEED = 0.75

CRIT = {"HARD": P.HARD_FILL, "FIRM": P.FIRM_FILL, "SOFT": P.SOFT_FALLBACK}
# what the objective loses per unit of lateness. Hard tasks dominate, which is
# why the number tracks the black more than it tracks the amount of black
CRIT_COST = {"HARD": 2.0, "FIRM": 1.2, "SOFT": 0.7}
CRIT_P = [("HARD", 0.30), ("FIRM", 0.30), ("SOFT", 0.40)]

LEGEND = [("crit", "HARD", "hard"), ("crit", "FIRM", "firm"),
          ("crit", "SOFT", "soft"), ("kernel", None, "kernel"),
          ("link", None, "link"), ("miss", None, "past its deadline")]

# ----------------------------- the attempts -------------------------------
N_ATTEMPTS = 44                # bars on the strip. Fewer, wider, and slower to
                               # flick past than a bar for every episode
EPISODES = 640_000             # 20000 steps x 32 task sets
SEED = 7


def _pick_crit(rng):
    r = rng.random()
    acc = 0.0
    for name, p in CRIT_P:
        acc += p
        if r <= acc:
            return name
    return CRIT_P[-1][0]


def _free(blocks, core, t0, t1):
    return not any(b["core"] == core and b["t"] < t1 and t0 < b["t"] + b["d"]
                   for b in blocks)


def make_attempt(seed, quality, perfect=False):
    """One allocation, invented but self consistent.

    quality runs from 0 to 1. Higher packs the lanes tighter, and misses both
    less often and by less. The penalty is summed from the blocks that come out,
    so what the viewer counts on screen is what the number says.

    Lateness shrinks in size as well as in frequency on purpose. Driving the
    probability alone to zero produces a tail of exact zeros, which on the strip
    is a row of empty slots rather than a curve coming down.
    """
    rng = random.Random(seed)
    slack = 1.0 - 0.85 * quality
    miss_p = 0.0 if perfect else 0.10 + 0.30 * (1.0 - quality)
    max_late = 3.0 + 9.0 * (1.0 - quality)
    blocks, penalty = [], 0.0

    for core in LANES:
        t = rng.uniform(0.0, 4.0) * slack
        while t < HORIZON - 14.0:
            t += rng.uniform(0.5, 7.0) * slack
            kern = rng.choice([3.0, 4.0])
            if t + kern + 10.0 > HORIZON:
                break
            blocks.append(dict(core=core, kind=rng.choice(["wake", "disp"]),
                               t=t, d=kern, crit=None, late=0.0))
            t += kern
            dur = min(rng.uniform(9.0, 25.0), HORIZON - t)
            crit = _pick_crit(rng)
            # the low end tracks quality too, so the bars follow the trend
            # instead of drowning it in noise
            late = (round(rng.uniform(0.45 * max_late, max_late), 1)
                    if rng.random() < miss_p else 0.0)
            late = min(late, dur - 2.0)
            blocks.append(dict(core=core, kind="run", t=t, d=dur, crit=crit,
                               late=late))
            penalty += late * CRIT_COST[crit]
            t += dur

    # one message across the link, if there is room for both ends. A dependency
    # that crosses devices is charged on the sender and on the receiver
    for _ in range(12):
        st = rng.uniform(12.0, 46.0)
        rt = st + rng.uniform(7.0, 16.0)
        if _free(blocks, "A1", st, st + 4.0) and _free(blocks, "B1", rt, rt + 5.0):
            blocks.append(dict(core="A1", kind="send", t=st, d=4.0, crit=None,
                               late=0.0))
            blocks.append(dict(core="B1", kind="rx", t=rt, d=5.0, crit=None,
                               late=0.0))
            break

    blocks.sort(key=lambda b: (b["t"], b["core"]))
    for i, b in enumerate(blocks):
        b["k"] = i
    return {"blocks": blocks, "penalty": int(round(penalty)), "n": len(blocks)}


def build_series(seed=SEED):
    """Every attempt in the video, worst to best. Only the last one is perfect.

    Quality stops short of 1 for all but the final attempt, so the curve settles
    low rather than pinning to the floor for the last third of the strip.
    """
    out = []
    for i in range(N_ATTEMPTS - 1):
        f = (i / (N_ATTEMPTS - 2)) ** 1.35
        out.append(make_attempt(seed * 1000 + i, 0.06 + 0.80 * f))
    out.append(make_attempt(seed * 1000 + N_ATTEMPTS - 1, 1.0, perfect=True))
    return out


# ----------------------------- drawing ------------------------------------
def xt(u):
    return GX + u * TS


def text_width(ax, artist):
    """Measured, not guessed. The legend words are not the same length."""
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    bb = artist.get_window_extent(renderer=renderer)
    (x0, _), (x1, _) = ax.transData.inverted().transform([(bb.x0, 0), (bb.x1, 0)])
    return x1 - x0


def swatch(ax, x, y, w, h, what, key):
    if what == "crit":
        gv.box(ax, x, y, w, h, CRIT[key], ec=INK, lw=1.0, z=6)
    elif what == "kernel":
        gv.box(ax, x, y, w, h, "none", ec=P.GAP, lw=1.3, hatch="////", z=6)
    elif what == "link":
        gv.box(ax, x, y, w, h, LINK_COL, ec=INK, lw=1.0, z=6)
    else:
        gv.box(ax, x, y, w, h, P.MISSED, ec=INK, lw=1.0, z=6)


def draw_legend(ax):
    """One centred row. Widths are measured first so the row can be centred."""
    probe, widths = [], []
    for _, _, label in LEGEND:
        t = ax.text(0, -50, label, fontsize=9.5)      # off screen, to measure
        widths.append(text_width(ax, t))
        probe.append(t)
    for t in probe:
        t.remove()
    total = sum(3.2 + 1.2 + w for w in widths) + 4.5 * (len(LEGEND) - 1)
    x = (100.0 - total) / 2.0
    for (what, key, label), w in zip(LEGEND, widths):
        swatch(ax, x, LEG_Y, 3.2, 2.6, what, key)
        ax.text(x + 4.4, LEG_Y + 1.3, label, ha="left", va="center",
                fontsize=9.5, color=INK)
        x += 3.2 + 1.2 + w + 4.5


def draw_lanes(ax, attempt, reveal, ok):
    for core, y in LANES.items():
        gv.box(ax, GX, y, GW, LANE_H, LANE_FILL[core], ec=P.HAIRLINE, lw=1.1, z=2)
        ax.text(GX - 1.2, y + LANE_H / 2, core, ha="right", va="center",
                fontsize=10, fontweight="bold", color=INK)
    ax.plot([GX, GX + GW], [RULE_Y, RULE_Y], color=LINK_COL, lw=1.6, zorder=3)
    ax.text(GX - 1.2, RULE_Y, "link", ha="right", va="center", fontsize=9,
            color=LINK_COL, fontweight="bold")

    if attempt is not None:
        n = attempt["n"]
        for b in attempt["blocks"]:
            f = 1.0 if reveal is None else float(
                np.clip((reveal - b["k"] / n) * n * 1.8, 0.0, 1.0))
            if f <= 0.01:
                continue
            y, x, w = LANES[b["core"]], xt(b["t"]), b["d"] * TS * f
            if b["kind"] == "run":
                fill = CRIT[b["crit"]]
                keep = (b["d"] - b["late"]) * TS * f
                gv.box(ax, x, y, min(w, keep), LANE_H, fill, ec=INK, lw=1.0, z=5)
                if w > keep:
                    gv.box(ax, x + keep, y, w - keep, LANE_H, P.MISSED, ec=INK,
                           lw=1.0, z=6)
                    dx = x + keep
                    ax.plot([dx, dx], [y - 0.8, y + LANE_H + 0.8], color=BAD,
                            lw=1.8, zorder=8)
            elif b["kind"] == "rx":
                gv.box(ax, x, y, w, LANE_H, LINK_COL, ec=INK, lw=1.0, z=5)
            elif b["kind"] == "send":
                gv.box(ax, x, y, w, LANE_H, "none", ec=LINK_COL, lw=1.5,
                       hatch="\\\\", z=5)
            else:
                gv.box(ax, x, y, w, LANE_H, "none", ec=P.GAP, lw=1.3,
                       hatch="////", z=5)

    if ok:
        ys = list(LANES.values())
        gv.box(ax, GX - 2.0, min(ys) - 1.6, GW + 4.0,
               max(ys) + LANE_H + 1.6 - (min(ys) - 1.6), "none", ec=GOOD,
               lw=2.0, z=10)


def draw_penalty(ax, value, grow):
    ax.text(PEN_X, PEN_LABEL_Y, "penalty", ha="center", va="center",
            fontsize=12, color=MUTED)
    if value is None:
        return
    col = GOOD if value == 0 else BAD
    size = 22.0 + 14.0 * gv.ease(grow)
    ax.text(PEN_X, PEN_Y, str(value), ha="center", va="center", fontsize=size,
            fontweight="bold", color=col)


def draw_strip(ax, pens, n_show, grow_last):
    ax.text(STRIP_X0, STRIP_TITLE_Y, "penalty, one bar per allocation",
            fontsize=11, fontweight="bold", color=MUTED)
    ax.plot([STRIP_X0, STRIP_X1], [STRIP_BASE] * 2, color=P.HAIRLINE, lw=1.2)
    if n_show <= 0:
        return

    pitch = (STRIP_X1 - STRIP_X0) / N_ATTEMPTS
    bw = pitch * 0.7
    # the ninetieth percentile, not the maximum. One unlucky task set otherwise
    # sets the scale and flattens every other bar against the axis
    top = max(sorted(pens)[int(0.9 * (len(pens) - 1))], 1) * 1.05
    tops = []
    for i in range(n_show):
        f = grow_last if i == n_show - 1 else 1.0
        h = min(pens[i] / top, 1.0) * STRIP_H * f
        x = STRIP_X0 + i * pitch
        last = i == n_show - 1
        gv.box(ax, x, STRIP_BASE, bw, max(h, 0.15),
               BAD if last else P.HAIRLINE, ec="none", lw=0, z=4)
        tops.append((x + bw / 2, STRIP_BASE + h))

    # the running mean, which is the column train.py prints while it works
    if n_show >= 5:
        xs, ys = [], []
        for i in range(n_show):
            lo = max(0, i - 7)
            m = sum(pens[lo:i + 1]) / (i - lo + 1)
            xs.append(tops[i][0])
            ys.append(STRIP_BASE + min(m / top, 1.0) * STRIP_H)
        ax.plot(xs, ys, color=ACCENT, lw=2.0, zorder=6)


def draw_count(ax, n_show):
    if n_show <= 0:
        return
    n = int(round(EPISODES * n_show / N_ATTEMPTS / 1000.0) * 1000)
    ax.text(STRIP_X1, COUNT_Y, f"{n:,} allocations", ha="right", va="center",
            fontsize=11, color=MUTED)


# ----------------------------- the beats -----------------------------------
def build(series, speed=SPEED):
    """The beats. `name` is for the console only. Nothing is written on screen."""
    pens = [a["penalty"] for a in series]
    spec = []

    def add(nf, name, **kw):
        spec.append((max(1, int(round(nf / speed))), name, kw))

    add(75, "empty platform", i=None, shown=0)
    add(100, "first allocation", i=0, reveal=0.62, pen_at=0.62, bar_at=0.80,
        shown=0)
    add(70, "second", i=1, reveal=0.50, pen_at=0.50, bar_at=0.72, shown=1)
    add(60, "third", i=2, reveal=0.40, pen_at=0.40, bar_at=0.62, shown=2)
    add(180, "the ramp", i="ramp", shown="ramp", count=True)
    add(100, "after training", i=len(series) - 1, reveal=0.55, pen_at=0.55,
        bar_at=0.55, shown=N_ATTEMPTS - 1, count=True, ok=True)
    return spec, pens


def render(ax, kw, e, series, pens):
    i = kw["i"]
    shown = kw["shown"]
    grow_last = 1.0
    reveal = None
    value = None
    grow = 1.0

    if i == "ramp":
        # attempts flick past. The strip is what the eye follows here
        step = int(np.clip(e * (N_ATTEMPTS - 4), 0, N_ATTEMPTS - 4 - 1))
        idx = 3 + step
        attempt = series[idx]
        shown = idx + 1
        value = attempt["penalty"]
    elif i is None:
        attempt = None
    else:
        attempt = series[i]
        reveal = min(e / kw["reveal"], 1.0)
        if e >= kw["pen_at"]:
            value = attempt["penalty"]
            grow = min((e - kw["pen_at"]) / 0.18, 1.0)
        if e >= kw["bar_at"]:
            shown = shown + 1
            grow_last = min((e - kw["bar_at"]) / 0.18, 1.0)

    draw_lanes(ax, attempt, reveal, kw.get("ok", False))
    draw_penalty(ax, value, grow)
    draw_strip(ax, pens, shown, grow_last)
    if kw.get("count"):
        draw_count(ax, shown)
    draw_legend(ax)


# ----------------------------- output --------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="learning.mp4")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--gif", action="store_true", help="also write a .gif")
    ap.add_argument("--series", action="store_true",
                    help="print the penalties and exit, without drawing")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="changes every attempt in the video. Worth a look with "
                         "--series before committing to one")
    ap.add_argument("--speed", type=float, default=SPEED,
                    help="below 1 is slower. Scales every beat, so the pacing "
                         "between them does not change")
    args = ap.parse_args()

    series = build_series(args.seed)
    spec, pens = build(series, args.speed)

    if args.series:
        top = max(max(pens), 1)
        for i, p in enumerate(pens):
            print(f"  {i:>3}  {p:>4}  {'#' * int(round(p / top * 50))}")
        print(f"\nfirst three {pens[:3]}   last {pens[-1]}   max {top}   "
              f"zeros {sum(1 for p in pens if p == 0)}")
        return

    total = sum(nf for nf, _, _ in spec)
    print(f"{len(spec)} beats, {total} frames, {total / args.fps:.1f}s "
          f"at speed {args.speed}")
    ffmpeg = gv.ffmpeg_exe()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    fig.patch.set_facecolor("white")
    tmp = tempfile.mkdtemp(prefix="learn_")
    size = bg = vbox = None
    n = 0
    try:
        for b, (nf, name, kw) in enumerate(spec):
            for f in range(nf):
                ax.clear()
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.axis("off")
                render(ax, kw, (f + 1) / nf, series, pens)
                fig.canvas.draw()
                im = Image.fromarray(
                    np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
                im.save(os.path.join(tmp, f"f{n:05d}.png"))
                if size is None:
                    size = im.size
                    bg = Image.new("RGB", size, im.getpixel((0, 0)))
                bb = (ImageChops.difference(im, bg).convert("L")
                      .point(lambda v: 255 if v > 8 else 0).getbbox())
                if bb:
                    vbox = bb if vbox is None else (
                        min(vbox[0], bb[0]), min(vbox[1], bb[1]),
                        max(vbox[2], bb[2]), max(vbox[3], bb[3]))
                n += 1
            print(f"  {name:<18} done, {n}/{total} frames")

        Wpx, Hpx = size
        x0, y0 = max(0, vbox[0] - PAD), max(0, vbox[1] - PAD)
        cw = min(Wpx, vbox[2] + PAD) - x0
        ch = min(Hpx, vbox[3] + PAD) - y0
        cw -= cw % 2
        ch -= ch % 2
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps),
             "-i", os.path.join(tmp, "f%05d.png"),
             "-vf", f"crop={cw}:{ch}:{x0}:{y0}",
             "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
            check=True)
        if args.gif:
            pal = os.path.join(tmp, "pal.png")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", args.out,
                            "-vf", "palettegen=max_colors=128", pal], check=True)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", args.out,
                            "-i", pal, "-lavfi",
                            "scale=iw/2:ih/2:flags=lanczos[x];[x][1:v]paletteuse",
                            str(Path(args.out).with_suffix(".gif"))], check=True)
        print(f"{args.out}  {cw}x{ch}")
    finally:
        plt.close(fig)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
