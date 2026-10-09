#!/usr/bin/env python3
"""
Task allocation visualisations -> MP4s, all identically framed.

  manual.mp4   hand-planned allocation, step by step, both cores exactly full
  bfd_wfd.mp4  BFD, fade to white, then WFD — both reject one task
  opt.mp4      BFD result in one step, then swaps until everything fits
  rl.mp4       the agent: one choice that looks wrong, and the reason for it
  shape.mp4    why the pieces do not have a fixed size in the real problem

Run:  python scripts/fig/figparts.py
      python scripts/fig/figparts.py --only rl
      python scripts/fig/figparts.py --only rl rl_why shape
      python scripts/fig/figparts.py --list

`--only` renders a subset, which is what you want while iterating. One caveat:
the horizontal crop is shared across everything rendered in a single run, so the
bars line up when the slides sit next to each other. A full run writes that crop
to video_crop.json and later `--only` runs reuse it, so do one full run once the
set is final.

Needs ffmpeg. If it is not on PATH:  pip install imageio-ffmpeg
(or set the FFMPEG environment variable to the ffmpeg.exe you have).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageChops
import numpy as np
import argparse
import subprocess
import tempfile
import shutil
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import palette as P

# ----------------------------- configuration -----------------------------
CAP = 100
TASKS = [("T1", 30), ("T2", 65), ("T3", 20), ("T4", 45), ("T5", 15), ("T6", 25)]
CORES = ["C1", "C2"]
# output file -> segments rendered into it, separated by a fade through white
VIDEOS = [("manual", ["manual"]),
          ("bfd_wfd", ["bfd", "wfd"]),
          ("opt", ["opt"]),
          ("rl", ["rl"]),
          ("shape", ["shape"])]

FPS = 30
DPI = 170
PAD = 6                    # px of margin kept around the content bounding box
GAP = 20                   # blank frames held between two segments
FADE = 0.4                 # seconds of fade out / fade in around that gap
SAVE_GIF = False           # set True to also write a .gif per video
CROP_FILE = "video_crop.json"

# hand-planned order: fill C1 exactly, then C2 exactly
MANUAL_PLAN = [("T2", "C1"), ("T3", "C1"), ("T5", "C1"),
               ("T4", "C2"), ("T1", "C2"), ("T6", "C2")]

# local-search moves tried in opt.mp4: (task_a, task_b)
SWAPS = [("T2", "T4"), ("T1", "T6"), ("T1", "T3")]

# The agent takes tasks in the order it is given, not sorted by size, and it
# reaches the same 100/100 the hand plan does.
#
# The whole video hangs on step 2. Best fit would drop T2 on C2, leaving 5, which
# is the tightest fit available and exactly what the rule is for. The agent puts
# it on C1 and leaves 35, which looks wasteful. Follow the greedy branch through
# and it strands T6 with 20 and 5 free on the two cores. The 70 the agent
# refused to fill on C2 is where T4 and T6 end up. One choice that looks wrong,
# and the reason for it arriving twenty seconds later.
RL_PLAN = [("T1", "C2"), ("T2", "C1"), ("T3", "C1"),
           ("T4", "C2"), ("T5", "C1"), ("T6", "C2")]
RL_DEVIATION = 1                # index into RL_PLAN: the step best fit disagrees with
RL_ALT = "C2"                   # where best fit would have put it


# --------------------------- the agent video ------------------------------
# A different picture from the other three on purpose: this one sits on its own
# slide and has to carry dependencies, a link between two devices, kernel
# overhead and a preemption, none of which a bin-packing bar can show. So the
# right hand side is a timeline, and the colours come from the project palette
# so it matches viz_alloc and the platform figures.
AG_TASKS = [("T1", "HARD", 20), ("T2", "HARD", 14), ("T3", "FIRM", 24),
            ("T4", "SOFT", 30), ("T5", "SOFT", 22), ("T6", "HARD", 10)]
AG_EDGES = [("T1", "T2"), ("T1", "T3"), ("T3", "T4")]
AG_NODE = {"T1": (5.0, 80.0), "T2": (23.0, 89.0), "T3": (23.0, 72.0),
           "T4": (23.0, 62.0), "T5": (5.0, 51.0), "T6": (5.0, 41.0)}
AG_LSCALE = 0.34               # left panel: axis units per time unit
AG_NODE_H = 4.6

# the two device boxes are held well apart so the link and its label live in
# clear space rather than on top of a core
AG_DEVICES = [("Device A", 64.0, 25.0, ["A1", "A2"]),
              ("Device B", 28.0, 25.0, ["B1", "B2"])]
AG_CORE_Y = {"A1": 78.0, "A2": 68.0, "B1": 42.0, "B2": 32.0}
AG_LANE_H = 5.5
AG_GX, AG_GW, AG_HORIZON = 50.0, 47.0, 112.0
AG_TS = AG_GW / AG_HORIZON

# hand-authored schedule. An illustration of what the model produces, not a
# recording of it, same as every other video here.
#
# `keep` marks a block that is already on screen when its group is revealed, so
# that T5 stays where it is and T6 cuts into it, rather than T5 vanishing and
# being redrawn.
AG_BLOCKS = {
    "a1_t1":    [dict(core="A1", kind="wake", t=0, d=4, task="T1"),
                 dict(core="A1", kind="run", t=4, d=20, task="T1")],
    "a1_send":  [dict(core="A1", kind="send", t=24, d=4, task="T1")],
    "a1_t2":    [dict(core="A1", kind="disp", t=28, d=3, task="T2"),
                 dict(core="A1", kind="run", t=31, d=14, task="T2")],
    "b1_rx":    [dict(core="B1", kind="rx", t=40, d=6, task="link")],
    "b2_t3":    [dict(core="B2", kind="wake", t=46, d=4, task="T3"),
                 dict(core="B2", kind="run", t=50, d=24, task="T3")],
    "b2_t4":    [dict(core="B2", kind="disp", t=74, d=3, task="T4"),
                 dict(core="B2", kind="run", t=77, d=30, task="T4")],
    "a2_whole": [dict(core="A2", kind="wake", t=0, d=4, task="T5"),
                 dict(core="A2", kind="run", t=4, d=22, task="T5")],
    "a2_split": [dict(core="A2", kind="wake", t=0, d=4, task="T5", keep=True),
                 dict(core="A2", kind="run", t=4, d=10, task="T5", keep=True),
                 dict(core="A2", kind="run", t=32, d=12, task="T5", from_t=14),
                 dict(core="A2", kind="pre", t=14, d=4, task="T6", delay=0.6),
                 dict(core="A2", kind="run", t=18, d=10, task="T6", delay=0.6),
                 dict(core="A2", kind="pre", t=28, d=4, task="T5", delay=0.6)],
}
AG_LINK = dict(src_core="A1", src_t=28, dst_core="B1", dst_t=40)

# swatch, label. Sits between the picture and the caption so nothing on the
# timeline needs a word of its own.
AG_LEGEND = [(3.0, "crit", "HARD", "HARD"), (17.0, "crit", "FIRM", "FIRM"),
             (31.0, "crit", "SOFT", "SOFT"),
             (45.0, "kernel", None, "kernel"),
             (59.0, "preempt", None, "preempt"),
             (74.0, "send", None, "send"),
             (86.0, "rx", None, "receive")]
AG_LEG_Y = 14.0
SLIDE = 0.6                    # share of a beat spent opening the room
PLACED_ALPHA = 0.3             # how faint a task goes once it has a core

# None = no title drawn (the slide already carries one)
TITLES = {"manual": None,
          "bfd": "Best Fit Decreasing (BFD)",
          "wfd": "Worst Fit Decreasing (WFD)",
          "opt": None,
          "rl": None,
          "shape": None}
RULE = {"bfd": "Best fit = smallest leftover",
        "wfd": "Worst fit = largest leftover"}
PICK = {"bfd": "tightest", "wfd": "most room"}

W = dict(TASKS)
COLORS = {"T1": "#4C78A8", "T2": "#54A24B", "T3": "#B279A2",
          "T4": "#F58518", "T5": "#E45756", "T6": "#79706E"}
INK = "#1F2933"
MUTED = "#7B8794"
TRACK = "#EDF0F4"
GOOD = "#2E7D32"
BAD = "#C62828"
ACCENT = "#5E35B1"         # local search

# layout in axis units (axes span 0..100 in both directions)
BIN_W = 40.0
UNIT = BIN_W / CAP
BAR_H = 7.5
QX = 6.0
QTOP = 72.0
QGAP = 10.4
BX = 52.0
BY = {"C1": 62.0, "C2": 45.0}
REJ_Y = 26.0

order = sorted(TASKS, key=lambda kv: -kv[1])
unsorted_slot = {n: i for i, (n, _) in enumerate(TASKS)}
sorted_slot = {n: i for i, (n, _) in enumerate(order)}


def ffmpeg_exe():
    """ffmpeg on PATH, or the copy bundled with imageio-ffmpeg."""
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise SystemExit(
            "ffmpeg not found.\n"
            "  pip install imageio-ffmpeg\n"
            "or set FFMPEG to the full path of ffmpeg.exe.")


# ----------------------------- helpers -----------------------------------
def load_of(bars):
    return sum(w for _, w in bars)


def state(queue, placed, rejected, slots):
    return {"queue": list(queue),
            "placed": {c: list(v) for c, v in placed.items()},
            "rejected": list(rejected),
            "slots": slots}


def positions(st):
    """Where every task sits in this state."""
    pos = {}
    for n in st["queue"]:
        pos[n] = (QX, slot_y(st["slots"][n]))
    for c in CORES:
        off = 0.0
        for n, w in st["placed"][c]:
            pos[n] = (BX + off, BY[c])
            off += w * UNIT
    off = 0.0
    for n, w in st["rejected"]:
        pos[n] = (BX + off, REJ_Y)
        off += w * UNIT + 2
    return pos


def slot_y(i):
    return QTOP - i * QGAP


def greedy(policy):
    free = {c: CAP for c in CORES}
    placed = {c: [] for c in CORES}
    rej = []
    for n, w in order:
        cands = [c for c in CORES if free[c] >= w]
        if not cands:
            rej.append((n, w))
            continue
        pick = min if policy == "bfd" else max
        b = pick(cands, key=lambda c: free[c] - w)
        free[b] -= w
        placed[b].append((n, w))
    return placed, rej


# ----------------------------- spec builders ------------------------------
def build_manual():
    spec = []

    def add(kind, nf, **kw):
        spec.append((kind, nf, kw))

    queue = [n for n, _ in TASKS]
    placed = {c: [] for c in CORES}
    st = state(queue, placed, [], unsorted_slot)
    add("static", 55, st=st, caption=f"{len(TASKS)} tasks  ·  {len(CORES)} cores",
        sub="Allocation planned by hand — all tasks considered together")

    for name, core in MANUAL_PLAN:
        w = W[name]
        prev = st
        queue = [n for n in queue if n != name]
        placed[core].append((name, w))
        st = state(queue, placed, [], unsorted_slot)
        add("move", 24, frm=prev, to=st, task=name,
            caption=f"Place {name} ({w}) on {core}")
        free = CAP - load_of(placed[core])
        add("settle", 13, st=st, task=name, core=core,
            caption=f"{name} allocated to {core}",
            sub=f"{core}: {free} free" if free else f"{core}: exactly full",
            sub_color=GOOD if not free else MUTED)

    loads = "  ·  ".join(f"{c} = {load_of(placed[c])}/100" for c in CORES)
    add("static", 80, st=st, caption=f"All {len(TASKS)} tasks allocated  ·  {loads}",
        sub="Nothing rejected, no idle time — this is the target",
        sub_color=GOOD)
    return spec


def build_greedy(policy):
    spec = []

    def add(kind, nf, **kw):
        spec.append((kind, nf, kw))

    queue = [n for n, _ in order]
    placed = {c: [] for c in CORES}
    rej = []
    st_un = state(queue, placed, rej, unsorted_slot)
    st = state(queue, placed, rej, sorted_slot)

    add("static", 40, st=st_un, caption=f"{len(TASKS)} tasks  ·  {len(CORES)} cores",
        sub="Bar length = execution time")
    add("sort", 36, frm=st_un, to=st,
        caption="Step 1  ·  sort tasks by decreasing execution time")
    add("static", 18, st=st, caption="Step 1  ·  sort tasks by decreasing execution time",
        sub="  ·  ".join(str(w) for _, w in order))

    for name, w in order:
        free = {c: CAP - load_of(placed[c]) for c in CORES}
        for c in CORES:
            add("probe", 21, st=st, task=name, core=c,
                fits=w <= free[c], free=free[c])
        cands = [c for c in CORES if free[c] >= w]

        if not cands:
            prev = st
            queue = [n for n in queue if n != name]
            rej = rej + [(name, w)]
            st = state(queue, placed, rej, sorted_slot)
            add("move", 24, frm=prev, to=st, task=name, bad=True,
                caption=f"No core can fit {name} ({w})",
                sub=f"{name} is rejected")
            free_txt = "  ·  ".join(f"{c}: {CAP - load_of(placed[c])} free"
                                    for c in CORES)
            add("static", 40, st=st, caption=f"{name} ({w}) rejected",
                sub=f"{free_txt}  —  enough time in total, but split across cores",
                sub_color=BAD, caption_color=BAD)
            continue

        pick = min if policy == "bfd" else max
        best = pick(cands, key=lambda c: free[c] - w)
        add("decide", 18, st=st, task=name, core=best,
            leftovers={c: free[c] - w for c in cands}, policy=policy)
        prev = st
        queue = [n for n in queue if n != name]
        placed[best].append((name, w))
        st = state(queue, placed, rej, sorted_slot)
        add("move", 20, frm=prev, to=st, task=name,
            caption=f"Allocate {name} to {best}")
        add("settle", 11, st=st, task=name, core=best,
            caption=f"{name} allocated to {best}",
            sub=f"{best}: {CAP - load_of(placed[best])} free")

    loads = "  ·  ".join(f"{c} = {load_of(placed[c])}/100" for c in CORES)
    idle = sum(CAP - load_of(placed[c]) for c in CORES)
    add("static", 85, st=st,
        caption=f"{policy.upper()} finishes  ·  {loads}",
        sub=(f"{len(rej)} task rejected — yet {idle} units of core time sit idle"
             if rej else "all tasks allocated"),
        sub_color=BAD if rej else GOOD)
    return spec


def build_opt():
    spec = []

    def add(kind, nf, **kw):
        spec.append((kind, nf, kw))

    placed, rej = greedy("bfd")
    stuck = rej[0][0]
    w_stuck = W[stuck]
    slots = unsorted_slot

    st0 = state([n for n, _ in TASKS], {c: [] for c in CORES}, [], slots)
    add("static", 40, st=st0, caption=f"{len(TASKS)} tasks  ·  {len(CORES)} cores",
        sub="Bar length = execution time")

    st = state([stuck], placed, [], slots)
    add("move", 32, frm=st0, to=st, task=None,
        caption="Step 0  ·  allocate with BFD")

    loads = "  ·  ".join(f"{c} = {load_of(placed[c])}/100" for c in CORES)
    add("static", 50, st=st, caption=f"BFD result  ·  {loads}",
        sub=f"{stuck} ({w_stuck}) is still waiting")

    for c in CORES:
        add("probe", 21, st=st, task=stuck, core=c,
            fits=False, free=CAP - load_of(placed[c]))

    prev = st
    st = state([], placed, [(stuck, w_stuck)], slots)
    add("move", 24, frm=prev, to=st, task=stuck, bad=True,
        caption=f"No core can fit {stuck} ({w_stuck})",
        sub=f"{stuck} is rejected")
    add("static", 34, st=st, caption="Greedy is out of moves — but we are not",
        sub="Try exchanging tasks between the cores", sub_color=ACCENT)

    for k, (a, b) in enumerate(SWAPS, start=1):
        core_a = next(c for c in CORES if any(n == a for n, _ in placed[c]))
        core_b = next(c for c in CORES if any(n == b for n, _ in placed[c]))
        add("propose", 30, st=st, pair=(a, b), cores=(core_a, core_b),
            caption=f"Trial {k}  ·  swap {a} ({W[a]}) ↔ {b} ({W[b]})")

        cand = {c: list(v) for c, v in placed.items()}
        cand[core_a] = [(b, W[b]) if n == a else (n, v) for n, v in cand[core_a]]
        cand[core_b] = [(a, W[a]) if n == b else (n, v) for n, v in cand[core_b]]
        prev = st
        st_c = state([], cand, [(stuck, w_stuck)], slots)
        add("move", 26, frm=prev, to=st_c, task=None,
            caption=f"Trial {k}  ·  swap {a} ↔ {b}")

        over = [c for c in CORES if load_of(cand[c]) > CAP]
        free = {c: CAP - load_of(cand[c]) for c in CORES}
        fits = [c for c in CORES if free[c] >= w_stuck and not over]

        if over:
            c = over[0]
            add("static", 46, st=st_c,
                caption=f"{c} is over capacity by {load_of(cand[c]) - CAP}",
                sub="Move rejected — undo it", sub_color=BAD, caption_color=BAD)
            add("move", 26, frm=st_c, to=st, task=None,
                caption=f"Trial {k}  ·  undo")
            continue

        if not fits:
            free_txt = "  ·  ".join(f"{c}: {free[c]} free" for c in CORES)
            add("static", 46, st=st_c, caption=f"Legal, but {free_txt}",
                sub=f"still no room for {stuck} ({w_stuck}) — undo it",
                sub_color=BAD, caption_color=INK)
            add("move", 26, frm=st_c, to=st, task=None,
                caption=f"Trial {k}  ·  undo")
            continue

        target = fits[0]
        free_txt = "  ·  ".join(f"{c}: {free[c]} free" for c in CORES)
        add("static", 40, st=st_c, caption=f"Legal  ·  {free_txt}",
            sub=f"{stuck} ({w_stuck}) fits on {target}", sub_color=GOOD)
        cand[target].append((stuck, w_stuck))
        st_f = state([], cand, [], slots)
        add("move", 26, frm=st_c, to=st_f, task=stuck,
            caption=f"Allocate {stuck} to {target}")
        loads = "  ·  ".join(f"{c} = {load_of(cand[c])}/100" for c in CORES)
        add("static", 85, st=st_f,
            caption=f"All {len(TASKS)} tasks allocated  ·  {loads}",
            sub="A few exchanges recover what the greedy pass threw away",
            sub_color=GOOD)
        break

    return spec


def build_rl():
    """The agent, on its own slide, in its own picture.

    Task graph on the left, coloured by criticality. Two devices on the right,
    each with two cores, drawn as timelines so the things a packing rule cannot
    see are visible: the kernel cost before every task, a dependency that stays
    on one core and costs nothing, one that crosses the link and costs on both
    sides, and a preemption that cuts a task in half.

    The expensive dependency comes first and the free one second, so the second
    reads as the contrast rather than as more of the same.
    """
    spec = []

    def add(nf, groups, new=(), **kw):
        blocks = []
        for g in groups:
            for blk in AG_BLOCKS[g]:
                b = dict(blk, new=(g in new) and not blk.get("keep"))
                if g not in new:
                    b.pop("from_t", None)      # it already slid, once
                    b.pop("delay", None)
                blocks.append(b)
        spec.append(("agent", nf, dict(blocks=blocks, **kw)))

    g = []
    add(80, g, placed=set(), caption="Six tasks with dependencies  ·  two devices, four cores",
        sub="every cost below the tasks is one a packing rule never sees")

    g = ["a1_t1"]
    add(55, g, new={"a1_t1"}, focus="T1", placed=set(), placing="T1",
        caption="T1 goes on core A1",
        sub="the kernel is charged before the task can start")

    g = g + ["a1_send", "b1_rx"]
    add(95, g, new={"a1_send", "b1_rx"}, focus="T3", edge=("T1", "T3"), link=True,
        placed={"T1"},
        caption="T3 depends on T1  ·  but it goes on the other device",
        sub="A1 pays to send it, 12 in flight, and B1 is interrupted to receive it")

    g = g + ["b2_t3"]
    add(55, g, new={"b2_t3"}, focus="T3", placed={"T1"}, placing="T3",
        caption="T3 can only start once the data has arrived",
        sub="one dependency, three separate costs, purely because of where it sits")

    g = g + ["a1_t2"]
    add(60, g, new={"a1_t2"}, focus="T2", edge=("T1", "T2"), edge_ok=True,
        placed={"T1", "T3"}, placing="T2",
        caption="T2 depends on T1 as well  ·  same core this time",
        sub="nothing to send, nothing to receive, nothing to wait for")

    g = g + ["b2_t4"]
    add(55, g, new={"b2_t4"}, focus="T4", edge=("T3", "T4"), edge_ok=True,
        placed={"T1", "T2", "T3"}, placing="T4",
        caption="T4 depends on T3  ·  same core again",
        sub="free, like T2 was")

    g = g + ["a2_whole"]
    add(50, g, new={"a2_whole"}, focus="T5",
        placed={"T1", "T2", "T3", "T4"}, placing="T5",
        caption="T5 on core A2",
        sub="nothing else is there, so it runs straight through")

    g = [x for x in g if x != "a2_whole"] + ["a2_split"]
    add(90, g, new={"a2_split"}, focus="T6",
        placed={"T1", "T2", "T3", "T4", "T5"}, placing="T6",
        caption="T6 is more urgent, and it lands while T5 is running",
        sub="T5 is cut in half and pays again to get the core back")

    add(95, g, done=True, placed={n for n, _, _ in AG_TASKS},
        caption="All six allocated  ·  every deadline met",
        sub="one pass, and it weighed all of this on the way")
    return spec


def build_shape():
    """Why the pieces do not have a fixed size once the real costs are in.

    No allocator here at all, and no new visual vocabulary: the extra cost is
    drawn with the same red dashed hatch that means overflow in bfd_wfd, because
    it is the same idea — core time consumed by something nobody asked for.

    This is a prologue. It belongs before the greedy video, so that when greedy
    fails the viewer already knows the pieces change size.
    """
    spec = []

    def add(nf, **kw):
        spec.append(("shape", nf, kw))

    # 1 · the same task, two different cores
    add(85, bars={"C1": [("T2", 65, "ghost")], "C2": [("T2", 45, "ghost")]},
        caption="The same task on two different cores",
        sub="65 here, 45 there — bin packing assumes it is one number")

    # 2 · the kernel takes a cut from every task
    add(60, bars={"C1": [("T2", 65, "solid"), ("__k1", 5, "extra")], "C2": []},
        caption="Starting a task is not free",
        sub="the kernel charges for the switch — every task, every time")
    add(75, bars={"C1": [("T2", 65, "solid"), ("__k1", 5, "extra"),
                         ("T3", 20, "solid"), ("__k2", 5, "extra")],
                  "C2": [("T4", 45, "solid"), ("__k3", 5, "extra"),
                         ("T5", 15, "solid"), ("__k4", 5, "extra"),
                         ("T6", 25, "solid"), ("__k5", 5, "extra")]},
        caption="Small on its own",
        sub="but a core carrying three of them pays it three times")

    # 3 · two tasks that talk to each other
    add(70, bars={"C1": [("T1", 30, "solid"), ("T3", 20, "solid")], "C2": []},
        arrow=("T1", "T3"),
        caption="T1 sends data to T3",
        sub="on the same core it costs nothing", sub_color=GOOD)
    add(90, bars={"C1": [("T1", 30, "solid"), ("__l1", 8, "extra")],
                  "C2": [("T3", 20, "solid"), ("__l2", 8, "extra")]},
        arrow=("T1", "T3"),
        caption="Split them across cores",
        sub="now both sides pay for the link, and the pieces changed size again",
        sub_color=BAD)
    add(85, bars={"C1": [("T1", 30, "solid"), ("__l1", 8, "extra")],
                  "C2": [("T3", 20, "solid"), ("__l2", 8, "extra")]},
        arrow=("T1", "T3"),
        caption="A rule written for fixed-size pieces cannot see any of this")
    return spec


# ----------------------------- drawing ------------------------------------
def box(ax, x, y, w, h, fc, ec=INK, lw=1.4, alpha=1.0, ls="-", z=3, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, alpha=alpha, linestyle=ls,
                           zorder=z, hatch=hatch))


def label_task(ax, x, y, w, name, z=5, color="white"):
    val = W[name]
    if w >= 7.0:
        ax.text(x + w / 2, y + BAR_H / 2, f"{name} · {val}", ha="center",
                va="center", color=color, fontsize=11, fontweight="bold", zorder=z)
    elif w >= 3.2:
        ax.text(x + w / 2, y + BAR_H / 2, name, ha="center", va="center",
                color=color, fontsize=10.5, fontweight="bold", zorder=z)
        ax.text(x + w / 2, y + BAR_H + 1.0, str(val), ha="center", va="bottom",
                color=INK, fontsize=10, zorder=z)
    else:
        ax.text(x + w / 2, y + BAR_H + 1.0, f"{name}·{val}", ha="center",
                va="bottom", color=INK, fontsize=10, fontweight="bold", zorder=z)


def draw_chrome(ax, mode, has_queue):
    if mode == "rl":
        return                      # the agent video draws its own panels
    if TITLES.get(mode):
        ax.text(50, 94, TITLES[mode], ha="center", va="center", fontsize=19,
                fontweight="bold", color=INK)
    if has_queue:
        ax.text(QX, 83, "Tasks", fontsize=12, fontweight="bold", color=MUTED)
    ax.text(BX, 83, "Cores", fontsize=12, fontweight="bold", color=MUTED)


def draw_tracks(ax, st, show_free=True, hide_free=None):
    for c in CORES:
        y = BY[c]
        box(ax, BX, y, BIN_W, BAR_H, TRACK, ec=INK, lw=2.0, z=2)
        ax.text(BX - 2.5, y + BAR_H / 2, c, ha="right", va="center",
                fontsize=13, fontweight="bold", color=INK)
        if not show_free or c == hide_free:
            continue
        load = load_of(st["placed"][c])
        if load > CAP:
            ov = (load - CAP) * UNIT
            ax.plot([BX + BIN_W, BX + BIN_W], [y - 2.5, y + BAR_H + 2.5],
                    color=BAD, lw=2.5, zorder=8)
            box(ax, BX + BIN_W, y - 0.6, ov, BAR_H + 1.2, "none", ec=BAD,
                lw=2.2, ls="--", z=9, hatch="///")
            ax.text(BX + BIN_W + ov / 2, y + BAR_H + 3.0,
                    f"over by {load - CAP}", ha="center", va="bottom",
                    fontsize=11, color=BAD, fontweight="bold")
        else:
            ax.text(BX + BIN_W + 1.5, y + BAR_H / 2, f"{CAP - load} free",
                    ha="left", va="center", fontsize=11,
                    color=GOOD if load == CAP else MUTED)
    if st["rejected"]:
        ax.text(BX - 2.5, REJ_Y + BAR_H / 2, "Rejected", ha="right", va="center",
                fontsize=12, fontweight="bold", color=BAD)


def draw_tasks(ax, st, pos=None, dim=None, highlight=()):
    pos = pos or positions(st)
    rej_names = {n for n, _ in st["rejected"]}
    for name, (x, y) in pos.items():
        w = W[name] * UNIT
        a = 0.35 if (dim and name != dim) else 1.0
        if name in rej_names:
            box(ax, x, y, w, BAR_H, "#F6DADA", ec=BAD, lw=1.8, hatch="///", z=4)
            ax.text(x + w + 1.5, y + BAR_H / 2, f"{name} · {W[name]}", ha="left",
                    va="center", fontsize=11, color=BAD, fontweight="bold")
            continue
        lw = 2.8 if name in highlight else (2.4 if name == dim else 1.4)
        ec = ACCENT if name in highlight else INK
        box(ax, x, y, w, BAR_H, COLORS[name], ec=ec, lw=lw, alpha=a, z=4)
        if abs(x - QX) < 1e-6:
            ax.text(x + w + 1.2, y + BAR_H / 2, f"{name} · {W[name]}", ha="left",
                    va="center", fontsize=11, color=INK, alpha=a)
        else:
            label_task(ax, x, y, w, name)


def caption(ax, main, sub=None, sub_color=MUTED, main_color=INK):
    ax.text(50, 13, main, ha="center", va="center", fontsize=14,
            color=main_color, fontweight="bold")
    if sub:
        ax.text(50, 5.5, sub, ha="center", va="center", fontsize=12.5,
                color=sub_color)


AG_CRIT = {"HARD": P.HARD_FILL, "FIRM": P.FIRM_FILL, "SOFT": P.SOFT_FALLBACK}
AG_LANE_FILL = {"A1": P.CPU_RAMP["Kryo_Gen6"][0], "A2": P.CPU_RAMP["Kryo_Gen6"][1],
                "B1": P.UNKNOWN_RAMP[0], "B2": P.UNKNOWN_RAMP[1]}
AG_LINK_COL = P.LINK_COLOUR["eth_10g"]
AG_DUR = {n: d for n, _, d in AG_TASKS}
AG_CR = {n: c for n, c, _ in AG_TASKS}


def ag_x(time_units):
    return AG_GX + time_units * AG_TS


def ag_swatch(ax, x, y, w, h, what, key):
    """One legend mark, drawn exactly the way the timeline draws it."""
    if what == "crit":
        box(ax, x, y, w, h, AG_CRIT[key], ec=INK, lw=1.0, z=6)
    elif what == "kernel":
        box(ax, x, y, w, h, "none", ec=P.GAP, lw=1.3, hatch="////", z=6)
    elif what == "preempt":
        box(ax, x, y, w, h, "none", ec=P.GAP, lw=1.3, hatch="...", z=6)
    elif what == "send":
        box(ax, x, y, w, h, "none", ec=AG_LINK_COL, lw=1.3, hatch="\\\\", z=6)
    else:
        box(ax, x, y, w, h, AG_LINK_COL, ec=INK, lw=1.0, z=6)


def draw_agent_legend(ax):
    for x, what, key, label in AG_LEGEND:
        ag_swatch(ax, x, AG_LEG_Y, 3.4, 3.0, what, key)
        ax.text(x + 4.4, AG_LEG_Y + 1.5, label, ha="left", va="center",
                fontsize=9.5, color=INK)


def draw_agent_left(ax, focus, edge, edge_ok, placed=(), placing=None, t=1.0):
    """The task set: bars sized by execution time, coloured by criticality."""
    ax.text(4.0, 95.5, "Task set", fontsize=12, fontweight="bold", color=MUTED)
    span = {}
    for name, crit, dur in AG_TASKS:
        x, y = AG_NODE[name]
        w = dur * AG_LSCALE
        span[name] = (x, y, w)
        # once a task is on a core it fades here, so the queue empties as the
        # timelines fill and you can see at a glance what is left
        if name == placing:
            a = 1.0 - (1.0 - PLACED_ALPHA) * ease(t)
        elif name in placed:
            a = PLACED_ALPHA
        else:
            a = 1.0
        box(ax, x, y, w, AG_NODE_H, AG_CRIT[crit],
            ec=GOOD if name == focus else INK,
            lw=2.6 if name == focus else 1.2, alpha=a, z=4)
        ax.text(x + w + 1.4, y + AG_NODE_H / 2, f"{name} · {dur}", ha="left",
                va="center", fontsize=10.5, color=INK, alpha=a,
                fontweight="bold" if name == focus else "normal")

    for a, b in AG_EDGES:
        xa, ya, wa = span[a]
        xb, yb, wb = span[b]
        hot = edge == (a, b)
        col = (GOOD if edge_ok else AG_LINK_COL) if hot else P.HAIRLINE
        if xb > xa + wa - 0.5:
            p0 = (xa + wa, ya + AG_NODE_H / 2)
            p1 = (xb, yb + AG_NODE_H / 2)
            rad = 0.18 if yb > ya else -0.18
        else:
            p0 = (xa + wa / 2, ya)
            p1 = (xb + wb / 2, yb + AG_NODE_H)
            rad = 0.0
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="->", color=col,
                                    lw=3.0 if hot else 1.6,
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=9 if hot else 3)


def draw_agent_right(ax, blocks, link, done, t):
    """Two devices, two cores each, drawn as timelines."""
    for label, y0, h, cores in AG_DEVICES:
        box(ax, 45.5, y0, 53.5, h, P.DEVICE_FILL, ec=P.HAIRLINE, lw=1.6, z=1)
        ax.text(47.0, y0 + h - 3.2, label, fontsize=11.5, fontweight="bold",
                color=MUTED)
        for c in cores:
            y = AG_CORE_Y[c]
            box(ax, AG_GX, y, AG_GW, AG_LANE_H, AG_LANE_FILL[c],
                ec=P.HAIRLINE, lw=1.2, z=2)
            ax.text(AG_GX - 1.2, y + AG_LANE_H / 2, c, ha="right", va="center",
                    fontsize=10.5, fontweight="bold", color=INK)
            if done:
                box(ax, AG_GX, y - 0.7, AG_GW, AG_LANE_H + 1.4, "none", ec=GOOD,
                    lw=1.8, z=6)

    # the link, in the clear band between the two device boxes
    top, bot = AG_DEVICES[0][1], AG_DEVICES[1][1] + AG_DEVICES[1][2]
    ax.annotate("", xy=(47.8, bot), xytext=(47.8, top),
                arrowprops=dict(arrowstyle="-", color=AG_LINK_COL, lw=2.6),
                zorder=3)
    ax.text(49.4, (top + bot) / 2, "link", ha="left", va="center", fontsize=10,
            color=AG_LINK_COL, fontweight="bold")

    e = ease(t)
    for blk in blocks:
        y = AG_CORE_Y[blk["core"]]
        full = blk["d"] * AG_TS
        if "from_t" in blk:
            # already on the core: it is pushed along, never redrawn
            f = min(e / SLIDE, 1.0)
            x = ag_x(blk["from_t"] + (blk["t"] - blk["from_t"]) * f)
            w = full
        else:
            d = blk.get("delay", 0.0)
            f = 0.0 if e <= d else (e - d) / (1.0 - d)
            w = full * (f if blk.get("new") else 1.0)
            x = ag_x(blk["t"])
        if w <= 0.05:
            continue
        if blk["kind"] == "run":
            fill = AG_CRIT[AG_CR[blk["task"]]]
            box(ax, x, y, w, AG_LANE_H, fill, ec=INK, lw=1.0,
                z=8 if "from_t" in blk else 5)
            if w >= 5.0:
                ax.text(x + w / 2, y + AG_LANE_H / 2, blk["task"], ha="center",
                        va="center", fontsize=10, fontweight="bold",
                        color=P.ink_on(fill),
                        zorder=9 if "from_t" in blk else 6)
        elif blk["kind"] == "rx":
            box(ax, x, y, w, AG_LANE_H, AG_LINK_COL, ec=INK, lw=1.0, z=5)
        elif blk["kind"] == "send":
            box(ax, x, y, w, AG_LANE_H, "none", ec=AG_LINK_COL, lw=1.6,
                hatch="\\\\", z=5)
        elif blk["kind"] == "pre":
            box(ax, x, y, w, AG_LANE_H, "none", ec=P.GAP, lw=1.4, hatch="...",
                z=5)
        else:
            # wake and dispatch: the kernel's cut, never the task's own work
            box(ax, x, y, w, AG_LANE_H, "none", ec=P.GAP, lw=1.4, hatch="////",
                z=5)

    if link:
        x0, y0 = ag_x(AG_LINK["src_t"]), AG_CORE_Y[AG_LINK["src_core"]]
        x1 = ag_x(AG_LINK["dst_t"])
        y1 = AG_CORE_Y[AG_LINK["dst_core"]] + AG_LANE_H
        e = ease(t)
        ax.annotate("", xy=(x0 + (x1 - x0) * e, y0 + (y1 - y0) * e),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=AG_LINK_COL, lw=2.6),
                    zorder=10)
        band = (AG_DEVICES[0][1] + AG_DEVICES[1][1] + AG_DEVICES[1][2]) / 2
        ax.text((x0 + x1) / 2 + 1.2, band, "12 in flight", ha="left",
                va="center", fontsize=10, color=AG_LINK_COL, fontweight="bold",
                zorder=10)

    ax.annotate("", xy=(AG_GX + AG_GW, 24.0), xytext=(AG_GX, 24.0),
                arrowprops=dict(arrowstyle="->", color=P.HAIRLINE, lw=1.4))
    ax.text(AG_GX, 22.6, "time", ha="left", va="top", fontsize=10, color=MUTED)


def draw_agent(ax, kw, t):
    draw_agent_left(ax, kw.get("focus"), kw.get("edge"), kw.get("edge_ok", False),
                    kw.get("placed", ()), kw.get("placing"), t)
    draw_agent_right(ax, kw["blocks"], kw.get("link", False),
                     kw.get("done", False), t)
    draw_agent_legend(ax)
    ax.text(50, 7.0, kw["caption"], ha="center", va="center", fontsize=14,
            color=INK, fontweight="bold")
    if kw.get("sub"):
        ax.text(50, 1.8, kw["sub"], ha="center", va="center", fontsize=12.5,
                color=GOOD if kw.get("done") else MUTED)


def ease(t):
    return t * t * (3 - 2 * t)


def render(ax, mode, kind, kw, t, has_queue):
    draw_chrome(ax, mode, has_queue)

    if kind == "static":
        st = kw["st"]
        draw_tracks(ax, st)
        draw_tasks(ax, st)
        caption(ax, kw["caption"], kw.get("sub"),
                kw.get("sub_color", MUTED), kw.get("caption_color", INK))
        return

    if kind in ("move", "sort"):
        frm, to = kw["frm"], kw["to"]
        e = ease(t)
        p0, p1 = positions(frm), positions(to)
        pos = {}
        for n in set(p0) | set(p1):
            a = p0[n] if n in p0 else p1[n]
            b = p1[n] if n in p1 else p0[n]
            pos[n] = (a[0] + (b[0] - a[0]) * e, a[1] + (b[1] - a[1]) * e)
        shown = to if e > 0.5 else frm
        draw_tracks(ax, shown, show_free=(kind == "sort"))
        moving = kw.get("task")
        rejn = {n for n, _ in to["rejected"]}
        for name, (x, y) in pos.items():
            w = W[name] * UNIT
            if name in rejn and (name != moving or e > 0.85):
                box(ax, x, y, w, BAR_H, "#F6DADA", ec=BAD, lw=1.8,
                    hatch="///", z=6)
                continue
            ec = BAD if (name == moving and kw.get("bad")) else INK
            lw = 2.2 if name == moving else 1.4
            box(ax, x, y, w, BAR_H, COLORS[name], ec=ec, lw=lw,
                z=7 if name == moving else 4)
            if abs(x - QX) < 1.5:
                ax.text(x + w + 1.2, y + BAR_H / 2, f"{name} · {W[name]}",
                        ha="left", va="center", fontsize=11, color=INK)
            else:
                label_task(ax, x, y, w, name, z=8)
        caption(ax, kw["caption"], kw.get("sub"),
                BAD if kw.get("bad") else MUTED,
                BAD if kw.get("bad") else INK)
        return

    if kind == "settle":
        st = kw["st"]
        draw_tracks(ax, st)
        draw_tasks(ax, st)
        x, y = positions(st)[kw["task"]]
        box(ax, x, y - 0.8, W[kw["task"]] * UNIT, BAR_H + 1.6, "none",
            ec=GOOD, lw=2.5, alpha=1.0 - t, z=8)
        caption(ax, kw["caption"], kw.get("sub"), kw.get("sub_color", MUTED))
        return

    if kind == "probe":
        st, name, c = kw["st"], kw["task"], kw["core"]
        w = W[name] * UNIT
        fits, free = kw["fits"], kw["free"]
        draw_tracks(ax, st, hide_free=None if fits else c)
        draw_tasks(ax, st, dim=name)
        y = BY[c]
        x0 = BX + load_of(st["placed"][c]) * UNIT
        pulse = 0.45 + 0.25 * np.sin(t * np.pi * 2.0)
        box(ax, x0, y, w, BAR_H, COLORS[name], ec=GOOD if fits else BAD,
            lw=2.2, alpha=pulse, ls="--", z=6)
        if not fits:
            ov = x0 + w - (BX + BIN_W)
            box(ax, BX + BIN_W, y - 0.6, ov, BAR_H + 1.2, "none", ec=BAD,
                lw=2.2, ls="--", z=7, hatch="///")
            ax.plot([BX + BIN_W, BX + BIN_W], [y - 2.5, y + BAR_H + 2.5],
                    color=BAD, lw=2.5, zorder=8)
            ax.text(BX + BIN_W + ov / 2, y + BAR_H + 3.2, "overflow",
                    ha="center", va="bottom", fontsize=11, color=BAD,
                    fontweight="bold")
            caption(ax, f"Try {name} ({W[name]}) on {c}",
                    f"✗  {W[name]} > {free} free  —  does not fit", BAD)
        else:
            caption(ax, f"Try {name} ({W[name]}) on {c}",
                    f"✓  fits  —  would leave {free - W[name]} free", GOOD)
        return

    if kind == "decide":
        st, name, c = kw["st"], kw["task"], kw["core"]
        draw_tracks(ax, st)
        draw_tasks(ax, st, dim=name)
        x0 = BX + load_of(st["placed"][c]) * UNIT
        box(ax, x0, BY[c], W[name] * UNIT, BAR_H, COLORS[name], ec=GOOD,
            lw=2.4, alpha=0.55, ls="--", z=6)
        lo = kw["leftovers"]
        detail = "   ".join(f"{k}: {v} left" for k, v in lo.items())
        if len(lo) == 1:
            sub = f"only {c} fits"
        elif len(set(lo.values())) == 1:
            sub = f"tie ({detail})  →  take {c}"
        else:
            sub = f"{detail}  →  {PICK[kw['policy']]} is {c}"
        caption(ax, f"{RULE[kw['policy']]}  →  {c}", sub, GOOD)
        return

    if kind == "propose":
        st = kw["st"]
        a, b = kw["pair"]
        draw_tracks(ax, st)
        draw_tasks(ax, st, highlight=(a, b))
        pos = positions(st)
        ax1 = pos[a][0] + W[a] * UNIT / 2
        ax2 = pos[b][0] + W[b] * UNIT / 2
        y1, y2 = pos[a][1], pos[b][1]
        hi = max(y1, y2)
        lo = min(y1, y2)
        ax.annotate("", xy=(ax2, lo + BAR_H + 0.5), xytext=(ax1, hi - 0.5),
                    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=2.4,
                                    connectionstyle="arc3,rad=0.35"),
                    zorder=9)
        caption(ax, kw["caption"], "exchange them between the cores", ACCENT)
        return

    # --- the agent -------------------------------------------------------
    if kind == "agent":
        draw_agent(ax, kw, t)
        return

    if kind == "shape":
        pos = {}
        for c in CORES:
            y = BY[c]
            box(ax, BX, y, BIN_W, BAR_H, TRACK, ec=INK, lw=2.0, z=2)
            ax.text(BX - 2.5, y + BAR_H / 2, c, ha="right", va="center",
                    fontsize=13, fontweight="bold", color=INK)
            off = 0.0
            for name, units, style in kw["bars"].get(c, []):
                w = units * UNIT
                if style == "extra":
                    # not the task's own work: core time consumed by something
                    # nobody asked for. Same red dashed hatch bfd_wfd uses for
                    # overflow, so it needs no legend of its own.
                    box(ax, BX + off, y - 0.6, w, BAR_H + 1.2, "none", ec=BAD,
                        lw=2.2, ls="--", z=7, hatch="///")
                elif style == "ghost":
                    box(ax, BX + off, y, w, BAR_H, COLORS[name], ec=INK,
                        lw=1.8, alpha=0.55, ls="--", z=6)
                    ax.text(BX + off + w / 2, y + BAR_H / 2, f"{name} · {units}",
                            ha="center", va="center", color=INK, fontsize=11,
                            fontweight="bold", zorder=8)
                else:
                    box(ax, BX + off, y, w, BAR_H, COLORS[name], ec=INK,
                        lw=1.4, z=4)
                    if w >= 7.0:
                        ax.text(BX + off + w / 2, y + BAR_H / 2,
                                f"{name} · {units}", ha="center", va="center",
                                color="white", fontsize=11, fontweight="bold",
                                zorder=5)
                if not name.startswith("__"):
                    pos[name] = (BX + off, y, w)
                off += w
        a = kw.get("arrow")
        if a and a[0] in pos and a[1] in pos:
            x1, y1, w1 = pos[a[0]]
            x2, y2, w2 = pos[a[1]]
            ax.annotate("", xy=(x2 + w2 / 2, y2 + BAR_H + 0.5),
                        xytext=(x1 + w1 / 2, y1 + BAR_H + 0.5),
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2.2,
                                        connectionstyle="arc3,rad=0.35"),
                        zorder=9)
        caption(ax, kw["caption"], kw.get("sub"), kw.get("sub_color", MUTED))
        return


# ----------------------------- drive --------------------------------------
BUILDERS = {"manual": build_manual,
            "opt": build_opt,
            "rl": build_rl,
            "shape": build_shape}


def build(mode):
    return BUILDERS.get(mode, lambda: build_greedy(mode))()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    choices=[v for v, _ in VIDEOS],
                    help="render just these, instead of all of them")
    ap.add_argument("--list", action="store_true", help="show the names and exit")
    args = ap.parse_args()

    if args.list:
        for v, segs in VIDEOS:
            print(f"  {v:<10} {' + '.join(segs)}")
        return

    wanted = [(v, s) for v, s in VIDEOS if not args.only or v in args.only]
    full_run = len(wanted) == len(VIDEOS)

    ffmpeg = ffmpeg_exe()
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=DPI)
    fig.patch.set_facecolor("white")

    dirs, cuts, boxes = {}, {}, {}
    size = bg = blank = None
    gx0 = gx1 = None

    for vid, segments in wanted:
        tmp = tempfile.mkdtemp(prefix=f"alloc_{vid}_")
        dirs[vid] = tmp
        cuts[vid] = []
        vbox = None
        n = 0
        for si, mode in enumerate(segments):
            if si:                                   # blank hold between segments
                cuts[vid].append(n)
                for _ in range(GAP):
                    blank.save(os.path.join(tmp, f"f{n:05d}.png"))
                    n += 1
            has_queue = mode != "shape"
            for kind, nf, kw in build(mode):
                for f in range(nf):
                    ax.clear()
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
                    ax.axis("off")
                    render(ax, mode, kind, kw, (f + 1) / nf, has_queue)
                    fig.canvas.draw()
                    im = Image.fromarray(
                        np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
                    im.save(os.path.join(tmp, f"f{n:05d}.png"))
                    if size is None:
                        size = im.size
                        bg = Image.new("RGB", size, im.getpixel((0, 0)))
                        blank = bg.copy()
                    b = (ImageChops.difference(im, bg).convert("L")
                         .point(lambda v: 255 if v > 8 else 0).getbbox())
                    if b:
                        vbox = b if vbox is None else (
                            min(vbox[0], b[0]), min(vbox[1], b[1]),
                            max(vbox[2], b[2]), max(vbox[3], b[3]))
                    n += 1
        # blend the frames around each cut toward white (chained ffmpeg fade
        # filters cannot do this: fade=out holds the faded state afterwards)
        nfade = max(1, int(round(FADE * FPS)))
        white = Image.new("RGB", size, (255, 255, 255))
        for c in cuts[vid]:
            for i in range(nfade):
                for idx, a in ((c - nfade + i, (i + 1) / nfade),
                               (c + GAP + i, 1.0 - (i + 1) / nfade)):
                    p = os.path.join(tmp, f"f{idx:05d}.png")
                    if os.path.exists(p):
                        Image.blend(Image.open(p).convert("RGB"), white, a).save(p)

        boxes[vid] = vbox
        gx0 = vbox[0] if gx0 is None else min(gx0, vbox[0])
        gx1 = vbox[2] if gx1 is None else max(gx1, vbox[2])
        print(f"{vid}: {n} frames ({n / FPS:.1f}s)")

    # Shared horizontal crop so the bars line up when the slides sit side by
    # side. A subset run reuses the crop a full run recorded, because deriving
    # it from a subset would shift the framing by a few pixels against videos
    # rendered earlier.
    if not full_run and os.path.exists(CROP_FILE):
        saved = json.load(open(CROP_FILE))
        gx0, gx1 = min(gx0, saved["gx0"]), max(gx1, saved["gx1"])
        print(f"reusing horizontal crop from {CROP_FILE}")
    elif full_run:
        json.dump({"gx0": gx0, "gx1": gx1}, open(CROP_FILE, "w"))

    Wpx, Hpx = size
    x0 = max(0, gx0 - PAD)
    cw = min(Wpx, gx1 + PAD) - x0
    cw -= cw % 2

    for vid, segments in wanted:
        vbox = boxes[vid]
        y0 = max(0, vbox[1] - PAD)
        ch = min(Hpx, vbox[3] + PAD) - y0
        ch -= ch % 2
        out = f"{vid}.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", os.path.join(dirs[vid], "f%05d.png"),
             "-vf", f"crop={cw}:{ch}:{x0}:{y0}",
             "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
        if SAVE_GIF:
            pal = os.path.join(dirs[vid], "pal.png")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", out,
                            "-vf", "palettegen=max_colors=128", pal], check=True)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", out,
                            "-i", pal, "-lavfi",
                            "scale=iw/2:ih/2:flags=lanczos[x];[x][1:v]paletteuse",
                            f"{vid}.gif"], check=True)
        shutil.rmtree(dirs[vid])
        print(f"{out}  {cw}x{ch}")


if __name__ == "__main__":
    main()
