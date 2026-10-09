#!/usr/bin/env python3
"""Build the thesis defense deck skeleton on the university template.

    python scripts/slides/gen_defense_deck.py

Writes out/slides/thesis_defense.pptx.

This is a skeleton. It places the things that are already fixed by the thesis
and leaves every sentence to the author:

  placed      titles, the chapter each slide draws on, figures from out/writing,
              videos from out/videos, the research questions, the response-time
              equation, the objective, the related-work table, the references
  left open   every explanatory sentence, as a marked grey box saying what
              belongs there

Nothing on a slide is written by this script except content that is already in
the thesis verbatim. Delete each grey box as you fill it.

Figures come from out/writing so the deck and the thesis cannot disagree. Run
`python scripts/fig/generate_figs.py all` first if any are missing.
"""

import argparse
import tempfile
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "out" / "writing"
VID = ROOT / "out" / "videos"
SLIDEFIG = ROOT / "out" / "slides"
TEMPLATE = ROOT / "reference" / "Template presentazione1.pptx"
OUT = ROOT / "out" / "slides" / "thesis_defense.pptx"

TITLE_LAYOUT, CONTENT_LAYOUT = 0, 2
PH_TITLE, PH_SUB = 0, 12

LEFT, TOP = Inches(0.30), Inches(1.30)
WIDTH, BOTTOM = Inches(9.40), Inches(7.10)

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x6E, 0x6C, 0x66)
HOLD_FILL = RGBColor(0xF2, 0xF2, 0xEF)
HOLD_LINE = RGBColor(0xBF, 0xBE, 0xB5)

_tmp = None


def tmpdir():
    global _tmp
    if _tmp is None:
        _tmp = tempfile.mkdtemp(prefix="deck_")
    return Path(_tmp)


def clear_slides(prs):
    ids = prs.slides._sldIdLst
    for sid in list(ids):
        prs.part.drop_rel(sid.rId)
        ids.remove(sid)


def slide(prs, title, chapter=None):
    s = prs.slides.add_slide(prs.slide_layouts[CONTENT_LAYOUT])
    keep = {PH_TITLE, PH_SUB, 13, 14}
    for ph in list(s.placeholders):
        idx = ph.placeholder_format.idx
        if idx == PH_TITLE:
            ph.text_frame.text = title
        elif idx == PH_SUB:
            if chapter:
                ph.text_frame.text = chapter
            else:
                ph._element.getparent().remove(ph._element)
        elif idx not in keep:
            ph._element.getparent().remove(ph._element)
    return s


def lines(s, items, top, left=LEFT, width=WIDTH, height=None, size=15,
          align=PP_ALIGN.LEFT, colour=INK, space=8):
    """Verbatim thesis content only: equations, research questions, references."""
    box = s.shapes.add_textbox(left, top, width, height or Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        it = (it,) if isinstance(it, str) else it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        p.line_spacing = 1.1
        r = p.add_run()
        r.text = it[0]
        r.font.size = Pt(it[1] if len(it) > 1 else size)
        r.font.bold = it[2] if len(it) > 2 else False
        r.font.color.rgb = it[3] if len(it) > 3 else colour
    return box


def hold(s, instruction, top, height, left=LEFT, width=WIDTH):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = HOLD_FILL
    shp.line.color.rgb = HOLD_LINE
    shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.12)
    for i, line in enumerate(instruction if isinstance(instruction, list)
                             else [instruction]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(2)
        r = p.add_run()
        r.text = line
        r.font.size = Pt(11)
        r.font.italic = True
        r.font.color.rgb = MUTED
    return top + height


def figure(s, name, top, max_h, left=LEFT, max_w=WIDTH):
    """out/writing first, then out/slides for the hand-made ones."""
    path = FIG / name
    if not path.exists():
        path = SLIDEFIG / name
    if not path.exists():
        return hold(s, f"figure not built: {name}", top, Inches(0.4), left, max_w)
    w_px, h_px = Image.open(path).size
    w, h = max_w, int(max_w * h_px / w_px)
    if h > max_h:
        h, w = max_h, int(max_h * w_px / h_px)
    s.shapes.add_picture(str(path), left + (max_w - w) // 2, top, width=w, height=h)
    return top + h


def poster(name):
    """A frame from the middle of the clip, so the slide does not show black."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(VID / name))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * 0.55)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    out = tmpdir() / f"{Path(name).stem}.png"
    cv2.imwrite(str(out), frame)
    return str(out)


def video(s, name, top, left, width, caption=None):
    path = VID / name
    if not path.exists():
        hold(s, f"video not found: out/videos/{name}", top, Inches(0.4),
             left, width)
        return top + Inches(0.4)
    p = poster(name)
    h = width * 9 // 16
    if p:
        w_px, h_px = Image.open(p).size
        h = int(width * h_px / w_px)
    s.shapes.add_movie(str(path), left, top, width, h,
                       poster_frame_image=p, mime_type="video/mp4")
    if caption:
        lines(s, [(caption, 12, False, MUTED)], top + h + Inches(0.05),
              left=left, width=width, align=PP_ALIGN.CENTER, space=0)
        h += Inches(0.3)
    return top + h


def notes(s, when, text):
    s.notes_slide.notes_text_frame.text = f"[{when}]\n\n{text}"


# --------------------------------------------------------------------------

def build(prs):
    # 1 ---------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[TITLE_LAYOUT])
    for ph in s.placeholders:
        i = ph.placeholder_format.idx
        if i == PH_TITLE:
            ph.text_frame.text = ("Overhead-Aware Reinforcement Learning for "
                                  "Real-Time Task Allocation on Heterogeneous "
                                  "Multicore Platforms")
        elif i == 10:
            ph.text_frame.text = "Mete Harun Akcay, MSc EDISS"
        elif i == 11:
            ph.text_frame.text = ("Supervisor: Prof. Fabrizio Rossi\n"
                                  "Company supervisor: Kimmo Hamynen")
    notes(s, "0:20", "")

    # 2 ---------------------------------------------------------------
    s = slide(prs, "Problem Statement")
    lines(s, [
        ("Input: a set of real-time tasks with periods, deadlines and data "
         "dependencies, and a description of the available hardware.", 15),
        ("Output: the device and the core that every task runs on.", 15),
        ("Scheduling inside a core is partitioned fixed priority. Priorities "
         "follow from the allocation, so the allocation is the only decision.",
         15),
    ], Inches(1.35), height=Inches(1.6), space=6)
    hold(s, ["PLACEHOLDER: the combinatorics table from the business deck, "
             "slide 2.",
             "6 tasks / 4 cores, 4^6 = 4096 cases, about 4 ms.",
             "12 / 8, 8^12 = 68.7 billion cases, about 19 hours.",
             "24 / 16, 16^24 = 7.9 x 10^28 cases, about 2.5 x 10^15 years."],
         Inches(3.10), Inches(1.9))
    lines(s, [
        ("Partitioned allocation is NP-hard even on identical processors [10], "
         "so the allocation has to be constructed rather than searched "
         "exhaustively.", 15),
    ], Inches(5.20), height=Inches(0.9))
    notes(s, "0:45", "")

    # 3 ---------------------------------------------------------------
    s = slide(prs, "Existing Solutions")
    w = Inches(2.95)
    for name, x, cap in (("manual.mp4", Inches(0.30), "Manual"),
                         ("bfd_wfd.mp4", Inches(3.52), "Greedy"),
                         ("opt.mp4", Inches(6.74), "Local search")):
        video(s, name, Inches(1.40), x, w, cap)
    for text, x in ((["+  Uses expert knowledge",
                      "-  Must be redone after changes"], Inches(0.30)),
                    (["+  Very fast and simple",
                      "-  Limited problem view"], Inches(3.52)),
                    (["+  Finds good solutions",
                      "-  Too slow for fast reaction"], Inches(6.74))):
        lines(s, [(ln, 13) for ln in text], Inches(4.35), left=x, width=w,
              height=Inches(0.9), space=3)
    lines(s, [
        ("All three decide placement without pricing what the hardware "
         "charges for it. Execution time differs by core, the kernel charges "
         "for every release and every preemption, and a message between "
         "devices costs a transmission delay plus processor time at both "
         "ends.", 14),
    ], Inches(5.40), height=Inches(1.1))
    notes(s, "0:55", "Play one clip, describe the other two.")

    # 4 ---------------------------------------------------------------
    s = slide(prs, "Complexity of the Real Problem")
    video(s, "complexity.mp4", Inches(1.35), Inches(0.55), Inches(5.1))
    lines(s, [
        ("What the decision has to account for:", 14, True),
        ("", 6),
        ("Task dependencies", 13),
        ("Communication delays between devices", 13),
        ("Core-specific execution and OS overheads", 13),
        ("Different task criticalities", 13),
        ("End-to-end timing constraints", 13),
        ("Different core types and roles", 13),
        ("Hardware failures", 13),
    ], Inches(1.45), left=Inches(5.95), width=Inches(3.75), height=Inches(3.0),
        space=5)
    lines(s, [
        ("A rule that cannot measure these costs cannot trade them against "
         "each other. The allocator needs the same timing model that judges "
         "the result.", 14),
    ], Inches(5.30), height=Inches(0.9))
    notes(s, "0:45", "")

    # 5 ---------------------------------------------------------------
    s = slide(prs, "Related Work")
    rows = [
        ("Peng et al. [36]", "Assignment and scheduling",
         "Communicating periodic tasks, distributed"),
        ("Schlatow et al. [40]", "Mapping, priorities, offsets",
         "Worst-case data age of control chains"),
        ("Cucinotta et al. [9]", "Placement, frequency, implementation",
         "DAG deadlines under partitioned EDF"),
        ("Panda [25]", "Learned priority ordering",
         "Response-time-analysis feedback"),
        ("SoCRATES [42]", "Learned ready-task placement",
         "Latency, run-time decisions"),
        ("Haouari et al. [17, 19]", "Q-learning placement and scheduling",
         "Per-instance, homogeneous assumptions"),
        ("This thesis", "Learned core allocation and repair",
         "Overhead-aware timing, routed communication"),
    ]
    tb = s.shapes.add_table(len(rows) + 1, 3, LEFT, Inches(1.35), WIDTH,
                            Inches(3.4)).table
    tb.columns[0].width, tb.columns[1].width, tb.columns[2].width = (
        Inches(2.2), Inches(3.3), Inches(3.9))
    for j, h in enumerate(("Work", "Optimised decision", "Timing scope")):
        c = tb.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
        c.text_frame.paragraphs[0].runs[0].font.bold = True
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = tb.cell(i, j)
            c.text = v
            r = c.text_frame.paragraphs[0].runs[0]
            r.font.size = Pt(11)
            r.font.bold = (i == len(rows))
    lines(s, [
        ("Existing work addresses these concerns separately. "
         "Communication-aware assignment optimises placement under "
         "transmission cost, chain-aware mapping optimises end-to-end "
         "latency, and learned methods address priority ordering or run-time "
         "dispatch.", 13),
        ("No existing method trains an allocation policy against a timing "
         "model that prices kernel overheads and routed communication "
         "together, and none addresses reallocation after a core failure "
         "within the same model.", 13),
    ], Inches(5.00), height=Inches(1.6), space=6)
    notes(s, "0:55", "")

    # 6 ---------------------------------------------------------------
    s = slide(prs, "Research Questions")
    lines(s, [
        ("RQ1   How does overhead-aware reinforcement learning compare with "
         "constructive heuristics in timing penalty and allocation time, and "
         "how does the starting allocation affect subsequent search?", 16),
        ("", 8),
        ("RQ2   How well does the trained RL allocator handle new task sets "
         "and hardware configurations without retraining?", 16),
        ("", 8),
        ("RQ3   After a core failure, how do repair and rebuild strategies "
         "compare in timing penalty and computation time?", 16),
    ], Inches(1.9), height=Inches(4.0))
    notes(s, "0:25", "Verbatim from Chapter 1.")

    # 7 ---------------------------------------------------------------
    s = slide(prs, "Thesis Architecture")
    y = figure(s, "architecture.drawio.png", Inches(1.35), Inches(4.3))
    lines(s, [
        ("Offline: data generation, policy training, and the timing model that "
         "supplies the training signal. Live: allocation, timing evaluation, "
         "optional refinement, and repair after a core failure. The timing "
         "model is the one component both paths share.", 13),
    ], y + Inches(0.15), height=Inches(1.0))
    notes(s, "0:45", "Sets up the order of everything that follows.")

    # 8 ---------------------------------------------------------------
    s = slide(prs, "System Model")
    figure(s, "example_taskset.png", Inches(1.35), Inches(2.3),
           left=Inches(0.25), max_w=Inches(4.6))
    figure(s, "example_platform.png", Inches(1.35), Inches(2.3),
           left=Inches(5.05), max_w=Inches(4.6))
    lines(s, [
        ("Five stored fields per task", 13),
        ("Criticality and weight derived from the period", 13),
        ("Edges are sampled, the consumer never waits", 13),
        ("Periods do not propagate along the graph", 13),
    ], Inches(3.80), left=Inches(0.25), width=Inches(4.6), height=Inches(1.6),
        space=6)
    lines(s, [
        ("Cores carry a CPU type and a core class", 13),
        ("Three classes, isolated, general and OS", 13),
        ("Core class selects the response-time tail factor", 13),
        ("One OS core per device, handling communication", 13),
    ], Inches(3.80), left=Inches(5.05), width=Inches(4.6), height=Inches(1.6),
        space=6)
    lines(s, [
        ("The allocation maps every task to one core. Priorities and routes "
         "follow from it, and no core is ever forbidden.", 13),
    ], Inches(5.55), height=Inches(0.8))
    notes(s, "0:50", "")

    # 9 ---------------------------------------------------------------
    s = slide(prs, "Overhead-Aware Timing Model")
    y = figure(s, "response_time.png", Inches(1.40), Inches(1.0),
               left=Inches(1.4), max_w=Inches(7.2))
    lines(s, [("alpha  core-class tail factor     w  wake     d  dispatch"
               "     rho  resume     k  preemption count", 12, False, MUTED)],
          y + Inches(0.10), align=PP_ALIGN.CENTER, height=Inches(0.35))
    lines(s, [
        ("w       the kernel cost of starting the task, charged once per "
         "analysed task", 13),
        ("C_eff   execution on the assigned CPU type, plus the CPU cost of "
         "sending each outgoing cross-device message", 13),
        ("C_h + d  every release of a higher-priority task costs its execution "
         "and one dispatch, because the kernel spends the switch whoever it "
         "switches to", 13),
        ("k(R)    preemptions. The first release of an interferer blocks the "
         "task, every later one preempts it and costs a resume", 13),
        ("alpha   core-class tail factor, p99.9 over p50, applied once to the "
         "whole window rather than to execution alone", 13),
        ("", 5),
        ("Chain latency adds, per hop, the link delay, one consumer period of "
         "sampling wait, and the consumer response time. The reported latency "
         "is the worst over all paths between the budget endpoints.", 13),
    ], y + Inches(0.45), height=Inches(2.9), space=5)
    notes(s, "1:15", "The slide that carries the technical credibility.")

    # 10 --------------------------------------------------------------
    s = slide(prs, "Allocation Objective")
    y = figure(s, "penalty.png", Inches(1.38), Inches(1.55),
               left=Inches(1.9), max_w=Inches(6.2))
    y = figure(s, "objective.png", y + Inches(0.12), Inches(0.55),
               left=Inches(2.6), max_w=Inches(4.8))
    y = figure(s, "penalty-example.png", y + Inches(0.18), Inches(1.75))
    lines(s, [
        ("Reported allocation quality is P_task + P_chain. The preference term "
         "is small enough that it only breaks ties between allocations that "
         "miss nothing. An allocation is perfect when both are zero.", 13),
    ], y + Inches(0.10), height=Inches(0.8))
    notes(s, "1:00", "The objective the policy is trained against and every "
          "method is scored on.")

    # 11 --------------------------------------------------------------
    s = slide(prs, "Learned Allocator")
    y = figure(s, "learned_policy_architecture.png", Inches(1.35), Inches(3.1))
    lines(s, [
        ("The policy places one task at a time. For the current task it scores "
         "every core, selects one, updates the state, and continues. Earlier "
         "placements are never revised.", 13),
        ("No feature names a device, a platform, a CPU type or a link type. A "
         "core is its numbers, so an unseen platform is a different set of "
         "numbers in the same slots.", 13),
        ("Training is 2,000 imitation updates from stored allocations, then "
         "REINFORCE against the objective above, with the policy's own "
         "deterministic rollout as the baseline.", 13),
    ], y + Inches(0.15), height=Inches(1.7), space=5)
    notes(s, "1:15", "")

    # 12 --------------------------------------------------------------
    s = slide(prs, "Benchmark and Experimental Design")
    lines(s, [
        ("Tasks", 14, True),
        ("Timing: WCET, deadline, period", 12),
        ("Periods: 100 us / 500 us / 1 ms / 2 ms", 12),
        ("Criticality: HARD / FIRM / SOFT, with task-specific weights", 12),
        ("Dependencies between tasks", 12),
        ("", 5),
        ("Task dataset", 14, True),
        ("16,000 generated task sets, 50-task catalogue", 12),
        ("9 to 86 tasks per set, 4 dependency levels", 12),
        ("Loads from 60% to 100%", 12),
    ], Inches(1.35), left=Inches(0.30), width=Inches(4.6), height=Inches(3.5),
        space=3)
    lines(s, [
        ("Platforms", 14, True),
        ("Devices: NXP i.MX95, Qualcomm QCS6490, Qualcomm IQ-9100", 12),
        ("CPU types (A55, A78, Kryo) and core roles (OS, general, isolated)",
         12),
        ("Links: USB / 1G Ethernet / 10G Ethernet, with delay and CPU cost",
         12),
        ("", 5),
        ("Platform dataset", 14, True),
        ("20 platform configurations, 800 task sets each", 12),
        ("1 to 4 devices, 6 to 28 cores", 12),
        ("5 configurations held out of training entirely", 12),
    ], Inches(1.35), left=Inches(5.05), width=Inches(4.65), height=Inches(3.5),
        space=3)
    lines(s, [
        ("Splits: 9,400 training, 200 validation, 2,400 unseen workloads on "
         "training platforms, 4,000 instances on the five held-out platforms.",
         13),
        ("Every method is evaluated on the same 6,400 held-out task sets and "
         "compared instance by instance. Each annealed variant receives the "
         "same move set, cooling schedule and budget of 4,000 candidate "
         "evaluations.", 13),
    ], Inches(5.00), height=Inches(1.5), space=5)
    notes(s, "0:50", "")

    # 13 --------------------------------------------------------------
    s = slide(prs, "Results: Constructive Allocation")
    y = figure(s, "onepass_quality.png", Inches(1.35), Inches(4.0))
    lines(s, [
        ("In one constructive pass the learned allocator reaches a mean "
         "timing penalty of 110.3, against 322.8 for worst fit, the strongest "
         "individual heuristic. Its median is 14.9 against 251.9, at 8.3 ms "
         "per instance against 47.4 ms to run all six heuristics.", 13),
        ("HEFT produces more perfect allocations, 25 percent against 20, so "
         "the two metrics rank the methods differently.", 13),
    ], y + Inches(0.12), height=Inches(1.5), space=5)
    notes(s, "0:55", "")

    # 14 --------------------------------------------------------------
    s = slide(prs, "Results: Refined Allocation")
    y = figure(s, "budget_curve.png", Inches(1.35), Inches(4.0))
    lines(s, [
        ("Every starting allocation receives the same move set, cooling "
         "schedule and budget. The learned start holds the lowest mean "
         "penalty at every measured budget, reaching 4.6 after 4,000 "
         "evaluations against 6.3 from best fit and 11.5 from HEFT.", 13),
        ("No heuristic reaches the one-pass learned result below about 74 "
         "evaluations. All seven runs take between 1.56 and 1.58 seconds, so "
         "the difference is not bought with time.", 13),
    ], y + Inches(0.12), height=Inches(1.5), space=5)
    notes(s, "1:05", "")

    # 15 --------------------------------------------------------------
    s = slide(prs, "Results: Generalisation")
    y = figure(s, "offline-generalisation.png", Inches(1.35), Inches(4.2))
    lines(s, [
        ("On 2,400 unseen workloads the refined learned allocation reaches "
         "3.7, and on 4,000 instances from five platforms excluded from "
         "training it reaches 5.1. The increase of 1.38 is the smallest of "
         "the seven methods, against 1.48 for best fit and 1.66 for worst "
         "fit. The heuristics also do worse on the held-out pool despite "
         "having no training set, so the increase cannot be attributed to "
         "transfer alone.", 13),
    ], y + Inches(0.12), height=Inches(1.2))
    notes(s, "0:35", "")

    # 16 --------------------------------------------------------------
    s = slide(prs, "Dynamic Allocation after Core Failure")
    y = figure(s, "repair-example.png", Inches(1.35), Inches(2.5))
    lines(s, [
        ("Repair preserves the assignments on the surviving cores during "
         "construction and reallocates only the displaced tasks.", 14),
        ("Rebuild constructs a new allocation for every task on the reduced "
         "platform.", 14),
        ("Both receive the same move set, cooling schedule and candidate "
         "evaluation budget, and both start from the same pre-failure "
         "allocation, so the comparison isolates the starting point.", 14),
    ], y + Inches(0.20), height=Inches(1.8), space=6)
    notes(s, "0:40", "")

    # 17 --------------------------------------------------------------
    s = slide(prs, "Results: Repair and Rebuild")
    y = figure(s, "repair-rebuild-budget-mean.png", Inches(1.35), Inches(4.2))
    lines(s, [
        ("Repair gives a lower mean penalty than rebuild for every allocator "
         "at every matched budget. With ten candidate evaluations the learned "
         "repair reaches 90.1 in 8.12 ms, against 139.4 in 11.8 ms for the "
         "learned rebuild. At 1,000 evaluations the two reach 21.5 and 27.4. "
         "Both perform the same number of evaluations, so the difference "
         "comes from the starting allocation.", 13),
    ], y + Inches(0.12), height=Inches(1.2))
    notes(s, "0:45", "")

    # 18 --------------------------------------------------------------
    s = slide(prs, "Limitations and Future Work")
    lines(s, [
        ("Hardware calibration is incomplete", 14, True),
        ("Measurements are done only on the i.MX95. Next: measure and "
         "calibrate all target boards and communication links.", 12),
        ("Limited number of algorithms to compare", 14, True),
        ("Six allocators are compared with the AI allocator. Next: include "
         "more algorithms for a more robust evaluation.", 12),
        ("Platform selection is outside the current scope", 14, True),
        ("The hardware platform is assumed given. Next: extend to be "
         "energy-aware, finding the lowest-power platform that admits an "
         "allocation and that allocation.", 12),
        ("Task descriptions are still manual", 14, True),
        ("Dependencies, deadlines and priorities must be provided beforehand. "
         "Next: extract them from source code with an LLM-based task "
         "descriptor.", 12),
    ], Inches(1.35), height=Inches(4.4), space=4)
    lines(s, [
        ("Methodological limits: the reward comes from the same timing model "
         "that scores the result, a tail factor does not give the response "
         "window a percentile guarantee, and the results come from one "
         "training run.", 12),
    ], Inches(5.90), height=Inches(1.0))
    notes(s, "0:40", "")

    # 19 --------------------------------------------------------------
    s = slide(prs, "Conclusions")
    lines(s, [
        ("RQ1", 15, True),
        ("In one pass the learned allocator reduces mean timing penalty by "
         "about 66 percent relative to the strongest heuristic, and it "
         "retains the lowest penalty at every matched search budget.", 14),
        ("", 6),
        ("RQ2", 15, True),
        ("It holds the lowest penalty on unseen workloads and on five "
         "platform configurations excluded from training, with the smallest "
         "relative increase between the two pools, and without retraining.",
         14),
        ("", 6),
        ("RQ3", 15, True),
        ("Over 12,800 single-core failures, repair gives a lower mean penalty "
         "than rebuild for every allocator at every matched budget, and the "
         "learned allocation is the best starting point to repair from.", 14),
        ("", 8),
        ("These results hold under the evaluated timing model and on the "
         "generated benchmark. They do not establish physical "
         "schedulability.", 13, False, MUTED),
    ], Inches(1.45), height=Inches(5.4), space=4)
    notes(s, "0:20", "")

    # 20 --------------------------------------------------------------
    s = slide(prs, "References")
    refs = [
        "[9] Cucinotta et al. Multi-Criteria Optimization of Real-Time DAGs "
        "on Heterogeneous Platforms under P-EDF. ACM TECS, 2023.",
        "[10] Davis and Burns. A Survey of Hard Real-Time Scheduling for "
        "Multiprocessor Systems. ACM Computing Surveys 43.4, 2011.",
        "[17] Haouari, Mzid and Mosbahi. A Two-Step Multi-objective "
        "Reinforcement Learning Optimization Approach. ENASE, 2026.",
        "[19] Haouari, Mzid and Mosbahi. Reinforcement Learning for "
        "Multi-Objective Task Placement on Heterogeneous Architectures. "
        "ENASE, 2024.",
        "[21] Joseph and Pandya. Finding Response Times in a Real-Time "
        "System. The Computer Journal 29.5, 1986.",
        "[23] Kirkpatrick, Gelatt and Vecchi. Optimization by Simulated "
        "Annealing. Science 220.4598, 1983.",
        "[24] Kool, van Hoof and Welling. Attention, Learn to Solve Routing "
        "Problems! ICLR, 2019.",
        "[25] Lee et al. Panda: Reinforcement Learning-Based Priority "
        "Assignment for Multi-Processor Real-Time Scheduling. IEEE Access 8, "
        "2020.",
        "[36] Peng, Shin and Abdelzaher. Assignment and Scheduling "
        "Communicating Periodic Tasks in Distributed Real-Time Systems. "
        "IEEE TSE, 1997.",
        "[40] Schlatow et al. Data-Age Analysis and Optimisation for "
        "Cause-Effect Chains in Automotive Control Systems. SIES, 2018.",
        "[42] Sung and Ryu. SoCRATES: System-on-Chip Resource Adaptive "
        "Scheduling using Deep Reinforcement Learning. ICMLA, 2021.",
        "[45] Topcuoglu, Hariri and Wu. Performance-Effective and "
        "Low-Complexity Task Scheduling for Heterogeneous Computing. "
        "IEEE TPDS 13.3, 2002.",
        "[46] Wilhelm et al. The Worst-Case Execution-Time Problem. "
        "ACM TECS 7.3, 2008.",
        "[47] Williams. Simple Statistical Gradient-Following Algorithms for "
        "Connectionist Reinforcement Learning. Machine Learning 8, 1992.",
    ]
    lines(s, [(r, 10) for r in refs], Inches(1.30), height=Inches(5.7),
          space=3)
    notes(s, "not timed", "Leave on screen during questions.")


def backup(prs):
    s = slide(prs, "Search Budget beyond the Reported Range")
    y = figure(s, "budget_curve_deep.png", Inches(1.35), Inches(4.1))
    lines(s, [
        ("Carried to 500,000 evaluations, every starting point converges to a "
         "mean penalty between 0.10 and 0.16 and a perfect rate between 94 "
         "and 96 percent. The learned start therefore reduces the search "
         "budget required rather than reaching a quality the heuristics "
         "cannot. 4,000 evaluations take about 1.6 seconds, 500,000 take "
         "about four minutes.", 13),
    ], y + Inches(0.12), height=Inches(1.2))
    notes(s, "backup", "If asked whether a longer search removes the advantage.")

    s = slide(prs, "Fairness of the Baselines")
    y = figure(s, "greedy_pair.png", Inches(1.35), Inches(2.2))
    lines(s, [
        ("Every heuristic evaluates candidate placements with the timing "
         "model that scores the final allocation, and its admission test "
         "reimplements the response-time equation exactly.", 14),
        ("All are communication-aware, and each has a criticality-ordered "
         "variant, so a win cannot be dismissed as a win over a weight-blind "
         "baseline.", 14),
        ("They are myopic rather than uninformed. Each placement follows a "
         "fixed local rule and is never revisited during construction.", 14),
    ], y + Inches(0.18), height=Inches(2.0), space=6)
    notes(s, "backup", "Most likely question about the comparison.")

    s = slide(prs, "Comparison with Published Learned Allocators")
    lines(s, [
        ("Panda [25] learns a priority ordering for global fixed-priority "
         "scheduling rather than a persistent core assignment.", 14),
        ("SoCRATES [42] learns run-time dispatch of ready tasks rather than a "
         "static mapping.", 14),
        ("Haouari et al. [17, 19] perform per-instance Q-learning under "
         "homogeneous task and platform assumptions.", 14),
        ("", 6),
        ("Each decides something other than a persistent task-to-core "
         "assignment, so none can be evaluated on this benchmark without "
         "altering its decision problem. Simulated annealing over the same "
         "evaluator, at matched budgets, represents per-instance search "
         "instead.", 14),
    ], Inches(1.5), height=Inches(3.4), space=6)
    notes(s, "backup", "")

    s = slide(prs, "Difficulty of the Benchmark")
    y = figure(s, "offline-difficulty.png", Inches(1.35), Inches(3.9))
    lines(s, [
        ("No instance in the benchmark is infeasible on capacity. Bounding "
         "every task by its cheapest possible core placement gives a maximum "
         "required utilisation of 0.930. At 4,000 evaluations, 82.8 percent "
         "of the held-out instances have a zero-penalty allocation on record. "
         "Load separates the methods clearly, coupling does not.", 13),
    ], y + Inches(0.12), height=Inches(1.2))
    notes(s, "backup", "")

    s = slide(prs, "Composition of the Residual Penalty")
    figure(s, "offline-behaviour.png", Inches(1.5), Inches(4.6))
    notes(s, "backup", "")

    s = slide(prs, "Heuristic Portfolio against the Learned Allocator")
    y = figure(s, "portfolio_rl_penalty_difference.png", Inches(1.35), Inches(4.1))
    lines(s, [
        ("Taking the best of all six heuristics per instance still leaves the "
         "learned allocator ahead by 30.9 on the mean, with a 95 percent "
         "bootstrap interval of 26.3 to 35.4. The portfolio needs six "
         "constructive passes at 47.4 ms against 8.3 ms, so it is not an "
         "equal-cost comparison.", 13),
    ], y + Inches(0.12), height=Inches(1.2))
    notes(s, "backup", "")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    tpl = Path(args.template)
    if not tpl.exists():
        raise SystemExit(f"no template at {tpl}")
    prs = Presentation(str(tpl))
    clear_slides(prs)
    build(prs)
    n_main = len(prs.slides._sldIdLst)
    if not args.no_backup:
        backup(prs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(out))
    except PermissionError:
        raise SystemExit(f"{out} is open in PowerPoint. Close it and rerun.")

    total = len(prs.slides._sldIdLst)
    holds = sum(1 for s in prs.slides for sh in s.shapes
                if sh.has_text_frame and "PLACEHOLDER" in sh.text_frame.text)
    print(f"wrote {out}")
    print(f"  {n_main} slides to present, {total - n_main} backup")
    print(f"  {holds} placeholders to fill. Delete each box as you fill it.")


if __name__ == "__main__":
    main()
