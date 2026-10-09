"""Gantt chart of one allocation, one lane per core.

    python scripts/visualization/viz_alloc.py 2_3_1_1 bfd          best-fit
    python scripts/visualization/viz_alloc.py 2_3_1_1 wfd          the same set, worst-fit
    python scripts/visualization/viz_alloc.py 2_3_1_1 sa_bfd       annealed from best-fit
    python scripts/visualization/viz_alloc.py 3_3_1_1 bfd -w 600   first 600 us only, easier to read

The platform comes from the task set, so it is never passed. Writes
out/allocations/<taskset>_<allocator>.png

The allocation is recomputed here rather than read back from
data/allocations, so an annealed allocator runs its whole search again and takes
a second or two. It is deterministic, so it lands on the same allocation the
saved run did. When the allocator searched, the title also reports the penalty of
the allocation it started from, which is what the picture should be compared
against.

Bars come from a simulated schedule, not from the response time equation. An
event driven fixed priority run, one job per core at any instant, everything
else genuinely waiting. Preemption is a result of the simulation, not an input.

Two things differ from the analytic model, and the picture says so:

  the tail multiplier inflates execution here, rather than the whole response
  time, because a bar has to be a length of time on a core

  a job that overruns its deadline keeps running, since the simulation has no
  reason to stop it, while the analytic model stops iterating at the deadline

So use this to see what an allocation does, and `timing.py` for the verdict.
"""

import argparse
import json
import sys
from collections import defaultdict
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import paths
import tasksets
from allocators import get, names
from palette import (GAP, HAIRLINE, HARD_FILL, FIRM_FILL, INK, INK_MUTED, MISSED,
                     SOFT_GRADIENT, SURFACE, crit_fill, ink_on, link_colour)
from timing import Model, priority_rank

# each kernel overhead gets its own hatch, so wake, dispatch and the switch
# back in after a preemption are told apart on sight
GAP_HATCH = {"wake": "...", "dispatch": "////", "preempt": "xxxx"}

# blank lanes left between two devices, so they cannot be mistaken for
# cores of the same chip. The link and its traffic are drawn in the band.
LINK_HATCH_COLOUR = "#00c000"   # link cpu cost, same green everywhere
DEVICE_GAP = 1.6

ROOT = Path(__file__).resolve().parent.parent.parent
EPS = 1e-9


def hyperperiod(periods):
    h = 1
    for p in periods:
        h = h * p // gcd(h, p)
    return h


def simulate(jobs, horizon, ov):
    """Event driven fixed priority. Returns blocks with start, dur, kind."""
    blocks, t, last = [], 0.0, None

    while t < horizon - EPS:
        ready = [j for j in jobs if j["release"] <= t + EPS and j["rem"] > EPS]
        if not ready:
            future = [j["release"] for j in jobs if j["rem"] > EPS and j["release"] > t]
            if not future:
                break
            blocks.append({"kind": "IDLE", "start": t, "dur": min(future) - t})
            t, last = min(future), None
            continue

        j = min(ready, key=lambda x: (x["prio"], x["release"]))

        if last is not j:
            kind, dur = (("wake", ov["wake"]) if last is None else
                         ("preempt", ov["preempt"]) if j["started"] else
                         ("dispatch", ov["dispatch"]))
            blocks.append({"kind": kind, "start": t, "dur": dur, "task": j["task"]})
            t += dur
            # something higher priority may have been released while the kernel
            # was switching, in which case it takes the core instead
            if any(x["prio"] < j["prio"] and x["rem"] > EPS and x["release"] <= t + EPS
                   for x in jobs):
                last = j
                continue

        # run until it finishes, or something with higher priority is released
        end = min(t + j["rem"], horizon)
        higher = [x["release"] for x in jobs
                  if x["prio"] < j["prio"] and x["rem"] > EPS and x["release"] > t]
        if higher:
            end = min(end, min(higher))

        if end - t > EPS:
            blocks.append({"kind": "RUN", "start": t, "dur": end - t,
                           "task": j["task"], "deadline_abs": j["deadline_abs"],
                           "rx": j["rx"]})
            j["rem"] -= end - t
            j["started"] = True
            t = end
        else:
            t += EPS * 10          # nothing progressed, do not spin
        last = j

    return blocks


def build_jobs(model, core_id, tids, c_eff, receives, horizon):
    tail = model.tail[model.core[core_id]["core_class"]]
    jobs = []

    # receive work first: an interrupt handler outranks every task on the core.
    # It has no deadline of its own, so it can never be drawn as late.
    for i, r in enumerate(receives):
        for k in range(int(horizon // r["period"])):
            jobs.append({"task": f"rx:{r['link_type']}", "prio": -1000 + i,
                         "release": float(k * r["period"]), "rem": r["c"] * tail,
                         "started": False, "deadline_abs": float("inf"),
                         "rx": r["link_type"]})

    prio = {t: i for i, t in enumerate(
        sorted(tids, key=lambda t: priority_rank(model.task, t)))}
    for tid in tids:
        t = model.task[tid]
        for k in range(int(horizon // t["period_us"])):
            rel = k * t["period_us"]
            jobs.append({"task": tid, "prio": prio[tid], "release": float(rel),
                         "rem": c_eff[tid] * tail, "started": False,
                         "deadline_abs": rel + t["deadline_us"], "rx": None})
    return jobs


def draw(model, res, alloc, taskset, platform, allocator_name, out, window=None,
         note=""):
    on_core = defaultdict(list)
    for tid, cid in alloc.items():
        on_core[cid].append(tid)
    rx = res["receives"]

    # Lanes grouped by device, main device first, so the two halves of a
    # crossing sit one above the other on a shared time axis. One empty lane's
    # worth of space between devices, and the link is drawn in it.
    order = [d["device_id"] for d in platform["devices"]]
    if len(order) > 2:
        leaves = order[1:]
        k = len(leaves) // 2
        order = leaves[:k] + [order[0]] + leaves[k:]

    lanes, groups, row_of, y = [], [], {}, 0.0
    for dev in order:
        here = [c for c in model.cores_of_device[dev] if on_core[c] or rx.get(c)]
        if not here:
            continue
        if lanes:
            y += DEVICE_GAP
        first = y
        for c in here:
            row_of[c] = y
            y += 1.0
        lanes += here
        groups.append((dev, first, y - 1.0))
    bottom = y - 1.0

    horizon = hyperperiod([model.task[t]["period_us"] for t in model.task])
    span = min(window, horizon) if window else horizon

    # Always simulate the whole hyperperiod, then just look at a window of it.
    # Simulating only the window would give a task slower than the window zero
    # jobs, and it would be reported as starved when it is merely off screen.
    per_core = {}
    for cid in lanes:
        cpu = model.core[cid]["cpu_type"]
        jobs = build_jobs(model, cid, on_core[cid],
                          {t: res["tasks"][t]["c_eff"] for t in on_core[cid]},
                          rx.get(cid, []), horizon)
        per_core[cid] = simulate(jobs, horizon, model.overhead[cpu])

    fig, ax = plt.subplots(figsize=(18, 0.95 * (bottom + 1) + 3.0), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, cid in enumerate(lanes):
        ax.add_patch(patches.Rectangle((0, row_of[cid] - 0.5), span, 1.0, zorder=0,
                                       facecolor="#f4f4f2" if i % 2 else SURFACE,
                                       edgecolor="none"))

    # share of each task's C_eff that is link cpu cost rather than its own work
    link_frac = {}
    for tid in model.task:
        base = model.exec_us[model.task[tid]["code"]][model.core[alloc[tid]]["cpu_type"]]
        ce = res["tasks"][tid]["c_eff"]
        link_frac[tid] = (ce - base) / ce if ce > 0 else 0.0

    # number tasks per core by first appearance, so a bar carries a short label
    numbering, n_late = {}, 0
    gaps = {"wake": 0.0, "dispatch": 0.0, "preempt": 0.0}
    for cid in lanes:
        seen = {}
        for b in per_core[cid]:
            if b["kind"] == "RUN" and not b["rx"] and b["task"] not in seen:
                seen[b["task"]] = len(seen) + 1
        numbering[cid] = seen

    for cid in lanes:
        row = row_of[cid]
        for b in per_core[cid]:
            if b["kind"] == "IDLE" or b["dur"] <= EPS or b["start"] >= span:
                continue
            s, w = b["start"], min(b["dur"], span - b["start"])

            if b["kind"] != "RUN":
                gaps[b["kind"]] += w
                ax.add_patch(patches.Rectangle(
                    (s, row - 0.30), w, 0.60, facecolor="none", edgecolor=GAP,
                    linewidth=0.6, hatch=GAP_HATCH[b["kind"]], zorder=3))
                continue

            if b["rx"]:
                # link receive work on an os core. No deadline, so it is never
                # drawn as late, but it does push everything else out.
                ax.add_patch(patches.Rectangle(
                    (s, row - 0.36), w, 0.72, facecolor=link_colour(b["rx"]),
                    edgecolor="white", linewidth=0.6, zorder=2))
                continue

            fill = crit_fill(model.task[b["task"]]["period_us"])
            ax.add_patch(patches.Rectangle(
                (s, row - 0.36), w, 0.72, facecolor=fill,
                edgecolor="white", linewidth=0.6, zorder=2))

            # the tail of the bar is the link cpu cost this task pays to send
            # across a link, not its own work. Drawn proportionally, so a block
            # cut short by preemption keeps the same ratio.
            frac = link_frac[b["task"]]
            if frac > 0:
                lw_, x0 = w * frac, s + w * (1 - frac)
                ax.add_patch(patches.Rectangle(
                    (x0, row - 0.36), lw_, 0.72, facecolor="none",
                    edgecolor=LINK_HATCH_COLOUR, linewidth=0.0, hatch="---",
                    zorder=2.4))
                ax.plot([x0, x0], [row - 0.36, row + 0.36],
                        color=LINK_HATCH_COLOUR, linewidth=1.1, zorder=2.45)

            if s + w > b["deadline_abs"] + EPS:
                n_late += 1
                m0 = max(s, b["deadline_abs"])
                ax.add_patch(patches.Rectangle(
                    (m0, row - 0.36), max(s + w - m0, span / 400.0), 0.72,
                    facecolor=MISSED, edgecolor="none", zorder=2.6))

            n = numbering[cid].get(b["task"])
            if n is not None and w / span * 18 * 120 > 11:
                ax.text(s + w / 2, row, str(n), ha="center", va="center",
                        fontsize=6.5, color=ink_on(fill), zorder=4)

    ax.set_xlim(0, span)
    ax.set_ylim(bottom + 0.7, -0.5)
    ax.set_yticks([row_of[c] for c in lanes])
    # exec is C_eff/period, execution only. eff adds each task's dispatch gap
    # and its preemption switches, tail applied: what the core actually spends.
    ax.set_yticklabels(
        [f"{c}\n{model.core[c]['core_class']}  "
         f"exec {res['cores'][c]:.2f} -> eff {res['cores_effective'][c]:.2f}"
         for c in lanes], fontsize=7.5)
    ax.set_xlabel("time (us)")
    ax.grid(axis="x", color=HAIRLINE, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    # device name beside each group, and the link drawn horizontally across the
    # empty band between two devices, which is the only place it can go without
    # pretending the two are on the same chip
    for dev, r0, r1 in groups:
        ax.text(-0.155, (r0 + r1) / 2, dev, transform=ax.get_yaxis_transform(),
                rotation=90, ha="center", va="center", fontsize=9,
                weight="bold", color=INK)

    direct = {frozenset((lk["dev_a"], lk["dev_b"])): lk for lk in platform["links"]}

    for (below, r0, _), (above, _, prev_r1) in zip(groups[1:], groups):
        band = (prev_r1 + r0) / 2
        lk = direct.get(frozenset((above, below)))
        if lk is None:
            continue                    # not neighbours on the star, no band link
        t = model.link_tbl[lk["link_type"]]
        col = link_colour(lk["link_type"])

        # vertical, because the message really does go from one device to the
        # other. Double headed: data crosses in both directions.
        ax.add_patch(patches.FancyArrowPatch(
            (span * 0.30, prev_r1 + 0.55), (span * 0.30, r0 - 0.55),
            arrowstyle="<|-|>", mutation_scale=16, linewidth=2.5,
            color=col, zorder=6))

        # link facts on the right of the arrow, the traffic on the left
        ax.text(span * 0.335, band,
                f"{above}   {lk['link_type']}   delay {t['delay_us']} us   "
                f"cpu {t['cpu_us']} us each end   {below}",
                ha="left", va="center", fontsize=8.5, color=col, zorder=6,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=col))

        moving = [f"{a}->{b}" for a, b in model.edges
                  if {model.device_of(a, alloc), model.device_of(b, alloc)}
                  == {above, below}]
        per_line = 4
        listing = "\n".join(
            "  ".join(moving[i:i + per_line]) for i in range(0, len(moving), per_line))
        ax.text(span * 0.265, band,
                f"{len(moving)} edge(s) cross here\n{listing}" if moving
                else "nothing crosses here",
                ha="right", va="center", fontsize=7, family="monospace",
                color=col, zorder=6)

    missed = {t for t, r in res["tasks"].items() if r["miss"]}

    def label(tid):
        return tid

    for cid in lanes:
        pairs = sorted(numbering[cid].items(), key=lambda kv: kv[1])
        lines = ["  ".join(f"{n} {label(t)}{'*' if t in missed else ''}"
                           for t, n in pairs[i:i + 2])
                 for i in range(0, len(pairs), 2)]

        # A task can be allocated here and never run at all, when receive
        # handlers take the whole hyperperiod. Calling that "(empty)" would hide
        # the most interesting thing on the lane.
        starved = [t for t in on_core[cid] if t not in numbering[cid]]
        if starved:
            lines.append("STARVED " + ", ".join(label(t) for t in starved))
        if not lines:
            lines = ["(no task allocated)"]

        ax.text(1.012, row_of[cid], "\n".join(lines),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.5, family="monospace",
                color="#b03030" if starved else INK_MUTED)

    crossings = sum(1 for a, b in model.edges
                    if model.device_of(a, alloc) != model.device_of(b, alloc))
    link_total = sum(
        link_frac[t] * res["tasks"][t]["c_eff"] * (span / model.task[t]["period_us"])
        for t in model.task)
    p = res["penalty"]
    ax.set_title(
        f"{taskset['taskset_id']}  {allocator_name}  on "
        f"{taskset['generator']['platform_id']}   "
        f"{len(model.task)} tasks / {len(lanes)} cores over {len(groups)} device(s)   "
        f"{crossings} of {len(model.edges)} edges cross a link   "
        f"hyperperiod {horizon} us\n"
        f"PENALTY {p['total']:.2f} = misses {p['misses']:.2f} + budgets "
        f"{p['budgets']:.2f} + preference {p['preference']:.2f}   "
        f"({len(missed)} task(s) charged, marked * on the right){note}\n"
        f"{n_late} job(s) run past their deadline in this simulation   "
        f"kernel time: wake {gaps['wake']:.0f} + dispatch {gaps['dispatch']:.0f}"
        f" + preempt {gaps['preempt']:.0f} us   "
        f"link cpu paid by senders: {link_total:.0f} us per hyperperiod",
        fontsize=9.5)

    handles = [patches.Patch(facecolor=HARD_FILL, label="HARD (100 us)"),
               patches.Patch(facecolor=FIRM_FILL, label="FIRM (500 us)"),
               patches.Patch(facecolor=SOFT_GRADIENT[1000], label="SOFT (1 ms)"),
               patches.Patch(facecolor=SOFT_GRADIENT[2000], label="SOFT (2 ms)"),
               patches.Patch(facecolor=MISSED, label="past deadline"),
               patches.Patch(facecolor="#cccccc", hatch="---", edgecolor=LINK_HATCH_COLOUR,
                             label="link cpu cost (sending)")]
    handles += [patches.Patch(facecolor=link_colour(lt), label=f"{lt} receive")
                for lt in sorted({r["link_type"] for rs in rx.values() for r in rs})]
    handles += [patches.Patch(facecolor="none", edgecolor=GAP, hatch=h, label=k)
                for k, h in GAP_HATCH.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=8, fontsize=8, frameon=False)

    # fixed strips in inches for the title and the legend, converted to the
    # fractions subplots_adjust wants. A percentage would shrink the plot on a
    # tall figure and crowd it on a short one.
    h = fig.get_figheight()
    fig.subplots_adjust(left=0.19, right=0.80,
                        top=1 - 0.95 / h, bottom=1.00 / h)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("taskset", help="231 for taskset_2_3_1")
    ap.add_argument("allocator", choices=names())
    ap.add_argument("-w", "--window", type=float, default=None,
                    help="show only the first N us of the hyperperiod")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    taskset = tasksets.load(args.taskset)
    platform = json.loads(paths.platform(
        taskset["generator"]["platform_id"]).read_text())
    model = Model(taskset, platform)
    model.platform_cores = platform["cores"]

    fn = get(args.allocator)
    if args.allocator.startswith("sa_"):
        print(f"{args.allocator} searches, this takes a moment...")
    alloc = fn(model, platform)
    model.alloc_of = alloc
    res = model.evaluate(alloc)

    # a search allocator leaves its statistics behind, and where it started from
    # is the only honest reference for the picture
    s = getattr(fn, "last", None)
    note = (f"   [search: started at {s['sa_start_penalty']:.2f}, "
            f"{s['sa_evals']} evaluations, {100 * s['sa_accept_rate']:.0f}% accepted]"
            if s else "")

    out = Path(args.out) if args.out else paths.out(
        "allocations", f"{taskset['taskset_id']}_{args.allocator}")
    draw(model, res, alloc, taskset, platform, args.allocator, out, args.window, note)


if __name__ == "__main__":
    main()
