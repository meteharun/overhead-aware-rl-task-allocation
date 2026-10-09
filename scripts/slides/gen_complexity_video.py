#!/usr/bin/env python3
"""How the problem grows -> complexity.mp4. Eight steps, about seventy seconds.

    python scripts/slides/gen_complexity_video.py
    python scripts/slides/gen_complexity_video.py --step 6      one step, as a png

Needs ffmpeg. If it is not on PATH:  pip install imageio-ffmpeg

Two tasks on one core, then one new thing per step until the picture is the real
problem. Every step opens with empty cores, moves the whole task set across at
once, and fades out before the next one.

A slower core is drawn as a shorter track rather than by inflating the tasks on
it, so a task keeps one width everywhere and the core is the thing that changes.

Same palette and primitives as the other videos.
"""

import argparse
import os
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

INK, MUTED = gv.INK, gv.MUTED
HAIR, LINK = P.HAIRLINE, P.LINK_COLOUR["eth_10g"]
FIGSIZE, DPI, FPS, PAD = (10, 5.8), 170, 30, 6

SIZE = {"T1": 18, "T2": 28, "T3": 12, "T4": 22, "T5": 16, "T6": 20}
COL = {"T1": "#4C78A8", "T2": "#F58518", "T3": "#54A24B", "T4": "#B279A2",
       "T5": "#E45756", "T6": "#72B7B2"}
KERNEL, PREEMPT, RX = 4, 5, 6

BAND_MID, ROW_H = 59.0, 12.0
SCALE = 0.42                         # one scale for both halves
TX, COL_W, NODE_H = 5.0, 14.0, 5.4
LX, LW = 56.0, 38.0                  # LW is the track of the fastest core
MAX_PITCH, DEV_GAP, BAND = 12.0, 2.0, 58.0
YCUT = 19.0                  # everything above it fades, the bar does not
LINK_COLS = [P.LINK_COLOUR["eth_10g"], P.LINK_COLOUR["usb"]]
BAR_Y, BAR_H = 10.5, 4.6
SETTING_Y = 23.5
RAMP = [(0.18, 0.49, 0.20), (0.85, 0.72, 0.13),
        (0.90, 0.49, 0.13), (0.78, 0.16, 0.16)]      # green, yellow, orange, red

HEAD_S, MOVE_S, TAIL_S = 4.0, 1.2, 4.0
STEP_S = HEAD_S + MOVE_S + TAIL_S
MOVE_FROM, MOVE_TO = HEAD_S / STEP_S, (HEAD_S + MOVE_S) / STEP_S
FADE = 9

#   cid, slowdown, class, cpu.  1.0 is the fastest core and gets the full track
A_FAST, B_SLOW, C_MID = 1.0, 1.35, 1.15


def _step(tasks, layout, edges, cores, alloc, **kw):
    return dict(tasks=tasks, layout=layout, edges=edges, cores=cores,
                alloc=alloc, **kw)


GRID4 = [("T1", 0, 0), ("T2", 1, 0), ("T3", 0, 1), ("T4", 1, 1)]
SIX = [("T1", 0, 0), ("T2", 1, 0), ("T5", 2, 0),
       ("T3", 0, 1), ("T4", 1, 1), ("T6", 2, 1)]
DEP6 = [("T1", "T2"), ("T3", "T2"), ("T2", "T5"),
        ("T3", "T4"), ("T4", "T5"), ("T4", "T6")]
K = "Kryo_Gen6"
T4S = ["T1", "T2", "T3", "T4"]
T6S = T4S + ["T5", "T6"]

DEV_A = ("Device A", [("A1", A_FAST, "os", K), ("A2", A_FAST, "isolated", K)])
DEV_B = ("Device B", [("B1", B_SLOW, "os", "A55"),
                      ("B2", B_SLOW, "isolated", "A55")])
DEV_C = ("Device C", [("C1", C_MID, "os", "A78"),
                      ("C2", C_MID, "general", "A78"),
                      ("C3", C_MID, "isolated", "A78")])

# Dependencies only appear once there is more than one device, because on a
# single device an edge costs nothing and the picture would not change.
STEPS = [
    _step(["T1", "T2"], [("T1", 0, 0), ("T2", 0, 1)], [],
          [("", [("C1", A_FAST, "general", K)])], {"C1": ["T1", "T2"]},
          same=True, setting="identical tasks  ·  single core"),
    _step(["T1", "T2", "T3"], [("T1", 0, 0), ("T2", 0, 1), ("T3", 0, 2)], [],
          [("", [("C1", A_FAST, "general", K)])], {"C1": ["T1", "T2", "T3"]},
          setting="different tasks  ·  single core"),
    _step(T4S, GRID4, [],
          [("", [("C1", A_FAST, "general", K), ("C2", A_FAST, "general", K)])],
          {"C1": ["T2", "T3"], "C2": ["T4", "T1"]},
          setting="different tasks  ·  identical cores"),
    _step(T4S, GRID4, [],
          [("", [("C1", A_FAST, "general", K), ("C2", A_FAST, "general", K)])],
          {"C1": ["T2", "T3"], "C2": ["T4", "T1"]}, overhead=True,
          setting="different tasks  ·  identical cores  ·  kernel overheads"),
    _step(T4S, GRID4, [],
          [("", [("C1", A_FAST, "general", K),
                 ("C2", B_SLOW, "general", "A55")])],
          {"C1": ["T2", "T3"], "C2": ["T4", "T1"]}, overhead=True,
          setting="different tasks  ·  unequal cores  ·  kernel overheads"),
    _step(T4S, GRID4, [], [DEV_A, DEV_B],
          {"A1": ["T3"], "A2": ["T2"], "B1": ["T4"], "B2": ["T1"]},
          overhead=True,
          setting="different tasks  ·  connected devices  ·  kernel overheads"),
    _step(T6S, SIX, DEP6, [DEV_A, DEV_B],
          {"A1": ["T3"], "A2": ["T2", "T1"], "B1": ["T5"], "B2": ["T4", "T6"]},
          overhead=True, rx={"A1": 1, "B1": 1},
          setting="dependent tasks  ·  multiple devices  ·  overheads and communication"),
    _step(T6S, SIX, DEP6, [DEV_A, DEV_B, DEV_C],
          {"A1": [], "A2": ["T2", "T1"], "B1": ["T5"], "B2": ["T4"],
           "C1": [], "C2": ["T3"], "C3": ["T6"]},
          overhead=True, rx={"A1": 2, "B1": 1, "C1": 1},
          setting="dependent tasks  ·  heterogeneous devices  ·  overheads and communication"),
]
HOLD = int(round(STEP_S * FPS))


# ----------------------------- geometry ------------------------------------
def node_pos(step):
    rows = max(r for _, _, r in step["layout"]) + 1
    top = (rows - 1) * ROW_H
    shift = BAND_MID - (NODE_H / 2 + top / 2)
    return {n: (TX + c * COL_W, top - r * ROW_H + shift, look(step, n)[0] * SCALE)
            for n, c, r in step["layout"]}


def _layout(step, pitch):
    h = min(pitch * 0.62, 7.5)
    ys, boxes, y = {}, [], 0.0
    for label, cores in step["cores"]:
        top = y
        for cid, *_ in cores:
            y -= pitch
            ys[cid] = y
        if label:
            w = max(LW / f for _, f, _, _ in cores) + 7.0
            boxes.append((label, LX - 4.5, y - 2.0, w, top - y + 2.0 + h))
        y -= DEV_GAP * pitch
    lo = min([min(ys.values())] + [b for _, _, b, _, _ in boxes])
    hi = max([v + h for v in ys.values()]
             + [b + hh for _, _, b, _, hh in boxes])
    return ys, boxes, h, lo, hi


def lane_pos(step):
    """Lanes centred on the band. Device boxes add padding the pitch does not
    know about, so the fit is measured and corrected once rather than solved."""
    groups = step["cores"]
    n = sum(len(cs) for _, cs in groups)
    pitch = min(BAND / max(n + DEV_GAP * (len(groups) - 1), 1), MAX_PITCH)
    ys, boxes, h, lo, hi = _layout(step, pitch)
    if hi - lo > BAND:
        ys, boxes, h, lo, hi = _layout(step, pitch * BAND / (hi - lo))
    shift = BAND_MID - (lo + hi) / 2
    return ({c: v + shift for c, v in ys.items()},
            [(l, x, b + shift, w, hh) for l, x, b, w, hh in boxes], h)


def slots(step, h):
    """Final (x, width) per task. A task keeps one width, the track shrinks."""
    ys, _, _ = lane_pos(step)
    out, deco = {}, []
    for cid, names in step["alloc"].items():
        x = LX
        for _ in range(step.get("rx", {}).get(cid, 0)):
            deco.append(("rx", x, ys[cid], RX * SCALE))
            x += RX * SCALE
        for i, n in enumerate(names):
            if step.get("overhead"):
                deco.append(("kernel", x, ys[cid], KERNEL * SCALE))
                x += KERNEL * SCALE
                if i:
                    deco.append(("pre", x, ys[cid], PREEMPT * SCALE))
                    x += PREEMPT * SCALE
            w = look(step, n)[0] * SCALE
            out[n] = (x, ys[cid], w)
            x += w
    return out, deco


def look(step, n):
    """Size and colour. Step one deliberately makes every task the same."""
    return (SIZE["T1"], COL["T1"]) if step.get("same") else (SIZE[n], COL[n])


def ramp(frac):
    """Green to yellow to orange to red."""
    seg = min(int(frac * (len(RAMP) - 1)), len(RAMP) - 2)
    u = frac * (len(RAMP) - 1) - seg
    a, b = RAMP[seg], RAMP[seg + 1]
    return tuple(ai + (bi - ai) * u for ai, bi in zip(a, b))


def bar(ax, x, y, w, h, kind, colour=None, alpha=1.0, z=5):
    if kind == "task":
        gv.box(ax, x, y, w, h, colour, ec=INK, lw=1.1, z=z, alpha=alpha)
    elif kind == "rx":
        gv.box(ax, x, y, w, h, LINK, ec=INK, lw=1.0, z=z, alpha=alpha)
    elif kind == "pre":
        gv.box(ax, x, y, w, h, "none", ec=P.GAP, lw=1.2, hatch="xx", z=z,
               alpha=alpha)
    else:
        gv.box(ax, x, y, w, h, "none", ec=P.GAP, lw=1.2, hatch="////", z=z,
               alpha=alpha)


def render(ax, idx, t):
    step = STEPS[idx]
    nodes = node_pos(step)
    ys, boxes, h = lane_pos(step)
    ends, deco = slots(step, h)
    e = gv.ease(float(np.clip((t - MOVE_FROM) / (MOVE_TO - MOVE_FROM), 0, 1)))
    step_of = {"os": 0, "general": 1, "isolated": 2}
    info = {cid: (f, k, c) for _, cs in step["cores"] for cid, f, k, c in cs}

    # no name beside the node. At rest the moving block sits exactly on top of
    # it and already carries the name inside, so a label here reads as a second
    # copy of the same thing
    for n, (x, y, w) in nodes.items():
        gv.box(ax, x, y, w, NODE_H, look(step, n)[1], ec=INK, lw=1.1, z=4,
               alpha=1.0 - 0.72 * e)
    for a, b in step["edges"]:
        xa, ya, wa = nodes[a]
        xb, yb, _ = nodes[b]
        ax.annotate("", xy=(xb - 0.6, yb + NODE_H / 2),
                    xytext=(xa + wa + 0.6, ya + NODE_H / 2),
                    arrowprops=dict(arrowstyle="-|>", color=HAIR, lw=1.6),
                    zorder=3)

    for label, x, y0, w, hh in boxes:
        gv.box(ax, x, y0, w, hh, P.DEVICE_FILL, ec=HAIR, lw=1.3, z=1)
        ax.text(x, y0 + hh + 1.0, label, fontsize=9.5, color=MUTED,
                va="bottom")
    # each link sits in the clear gap between the two boxes it joins, with its
    # own colour, so two devices wired differently look different
    for i in range(len(boxes) - 1):
        top = boxes[i][2]
        bot = boxes[i + 1][2] + boxes[i + 1][4]
        x = LX + 7.0 + i * 9.0
        ax.plot([x, x], [bot, top], color=LINK_COLS[i % len(LINK_COLS)],
                lw=2.6, zorder=6)

    for cid, y in ys.items():
        f, klass, cpu = info[cid]
        gv.box(ax, LX, y, LW / f, h, P.CPU_RAMP[cpu][step_of[klass]], ec=HAIR,
               lw=1.1, z=2)
        ax.text(LX - 1.2, y + h / 2, cid, ha="right", va="center", fontsize=9,
                fontweight="bold", color=INK)

    for kind, x, y, w in deco:
        bar(ax, x, y, w * e, h, kind, alpha=e)

    for n, (ex, ey, ew) in ends.items():
        sx, sy, sw = nodes[n]
        x, y = sx + (ex - sx) * e, sy + (ey - sy) * e
        hh = NODE_H + (h - NODE_H) * e
        colour = look(step, n)[1]
        bar(ax, x, y, ew, hh, "task", colour, z=8)
        if ew >= 4.5:
            ax.text(x + ew / 2, y + hh / 2, n, ha="center", va="center",
                    fontsize=min(8.5, hh * 2.2), fontweight="bold",
                    color=P.ink_on(colour),
                    zorder=9)

    # a dependency that crosses a device boundary, drawn the way rl.mp4 draws it
    dev = {cid: g for g, (_, cs) in enumerate(step["cores"]) for cid, *_ in cs}
    on = {n: cid for cid, ns in step["alloc"].items() for n in ns}
    for a, b in step["edges"]:
        if a not in on or b not in on or dev[on[a]] == dev[on[b]]:
            continue
        ax_, ay, aw = ends[a]
        bx, by, _ = ends[b]
        ax.annotate("", xy=(bx, by + h / 2), xytext=(ax_ + aw, ay + h / 2),
                    arrowprops=dict(arrowstyle="->", color=LINK, lw=1.8,
                                    alpha=0.45 * e,
                                    connectionstyle="arc3,rad=0.12"),
                    zorder=10)

    # step number, and a bar that fills and reddens
    frac = (idx + 1) / len(STEPS)
    col = ramp(idx / (len(STEPS) - 1))
    ax.text(6.0, SETTING_Y, f"#{idx + 1}", fontsize=12,
            fontweight="bold", color=INK)
    ax.text(11.5, SETTING_Y, step["setting"], fontsize=12, color=INK)
    ax.text(6.0, BAR_Y + BAR_H + 1.8, "complexity", fontsize=10, color=MUTED)
    gv.box(ax, 6.0, BAR_Y, 88.0, BAR_H, "white", ec=HAIR, lw=1.1, z=2)
    gv.box(ax, 6.0, BAR_Y, 88.0 * frac, BAR_H, col, ec="none", lw=0, z=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="complexity.mp4")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--hold", type=int, default=HOLD)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    fig.patch.set_facecolor("white")

    def frame(idx, t):
        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        render(ax, idx, t)

    if args.step:
        import paths
        frame(args.step - 1, 1.0)
        out = paths.out("writing", f"complexity_{args.step}")
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        print(f"wrote {out}")
        return

    ffmpeg = gv.ffmpeg_exe()
    total = len(STEPS) * args.hold
    print(f"{len(STEPS)} steps, {total} frames, {total / args.fps:.1f}s")
    tmp = tempfile.mkdtemp(prefix="cx_")
    size = bg = vbox = None
    n = 0
    try:
        for idx in range(len(STEPS)):
            for f in range(args.hold):
                frame(idx, (f + 1) / args.hold)
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
            print(f"  step {idx + 1} done")

        # the complexity bar is not part of the fade, so only the rows above
        # it are blended. The cut is taken from the axes, not guessed
        cut = int(round(size[1] - ax.transData.transform((0, YCUT))[1]))
        white = Image.new("RGB", (size[0], cut), (255, 255, 255))
        for idx in range(len(STEPS)):
            base = idx * args.hold
            for i in range(FADE):
                ends_ = [(base + args.hold - 1 - i, 1.0 - (i + 1) / FADE)]
                if idx:                       # the first step opens plainly
                    ends_.append((base + i, 1.0 - (i + 1) / FADE))
                for j, a in ends_:
                    p = os.path.join(tmp, f"f{j:05d}.png")
                    im = Image.open(p).convert("RGB")
                    im.paste(Image.blend(im.crop((0, 0, size[0], cut)),
                                         white, a), (0, 0))
                    im.save(p)

        W, H = size
        x0, y0 = max(0, vbox[0] - PAD), max(0, vbox[1] - PAD)
        cw = (min(W, vbox[2] + PAD) - x0) // 2 * 2
        ch = (min(H, vbox[3] + PAD) - y0) // 2 * 2
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.fps),
             "-i", os.path.join(tmp, "f%05d.png"),
             "-vf", f"crop={cw}:{ch}:{x0}:{y0}", "-an", "-c:v", "libx264",
             "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", args.out], check=True)
        print(f"{args.out}  {cw}x{ch}")
    finally:
        plt.close(fig)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
