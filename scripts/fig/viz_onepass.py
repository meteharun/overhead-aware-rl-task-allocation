#!/usr/bin/env python3
"""One-pass allocation quality, as a single bar chart.

    python scripts/fig/viz_onepass.py
    python scripts/fig/viz_onepass.py --dir slides

Writes out/writing/onepass_quality.png.

No allocator here has searched. Every bar is one constructive pass, so the
comparison is between placement rules rather than between budgets.

Quality only. What each one costs is reported separately, where a time axis can show a
ratio instead of printing a number under a label.

Portfolio is not an allocator. It is the per-instance minimum over the six
heuristics, so it is what running all of them and keeping the best would give.
The algorithm selection literature calls this the virtual best solver. It
belongs on the chart because it is the option available without a learned
policy, and omitting it overstates the gap.
"""

import argparse
import json
import statistics
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

import figstyle as F
from figstyle import INK, MUTED
from palette import SURFACE

ROOT = Path(__file__).resolve().parent.parent.parent
ALLOC = ROOT / "data" / "allocations"

FIGSIZE = (9.6, 5.2)
DPI = 300
PAD = 18
OUTDIR = "writing"

# None is a gap: the six heuristics, then what running all six would give, then
# the policy. The subject goes last so the chart ends on it
BARS = ["bfd", "bfd_crit", "wfd", "wfd_crit", "heft", "heft_crit",
        None, "best_greedy", None, "rl"]
PORTFOLIO = ("bfd", "bfd_crit", "wfd", "wfd_crit", "heft", "heft_crit")

TICK = {"bfd": "BFD", "bfd_crit": "BFD-C", "wfd": "WFD", "wfd_crit": "WFD-C",
        "heft": "HEFT", "heft_crit": "HEFT-C", "rl": "RL",
        "best_greedy": "Portfolio"}


def read(name):
    f = ALLOC / f"{name}.jsonl"
    if not f.exists():
        return {}
    out = {}
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("set") not in (None, "eval"):
                continue
            s = r["summary"]
            out[r["taskset_id"]] = (s["penalty_hard"], s.get("alloc_ms") or 0.0)
    return out


def gather(stat="mean"):
    """Penalty and time per allocator, over the shared instances.

    `stat` aggregates the per-instance values, and the portfolio is formed before
    that: its value on an instance is the best of the six there, so the median
    portfolio is the median of those per-instance minima, not the minimum of the
    six medians. The second would be a different and much weaker object.
    """
    agg = statistics.median if stat == "median" else         (lambda v: sum(v) / len(v))
    got = {n: read(n) for n in set(BARS) - {None}}
    got = {n: v for n, v in got.items() if v}
    missing = [n for n in BARS if n and n not in got and n != "best_greedy"]
    if missing:
        raise SystemExit("no allocations for " + ", ".join(missing)
                         + "\nrun: python scripts/run/run_allocators.py --eval")

    members = [got[m] for m in PORTFOLIO if m in got]
    if len(members) > 1:
        ids = set.intersection(*(set(m) for m in members))
        # the portfolio takes the best penalty of the six and pays for all six,
        # because all six had to run before the best of them was known
        got["best_greedy"] = {
            t: (min(m[t][0] for m in members), sum(m[t][1] for m in members))
            for t in ids}

    common = set.intersection(*(set(v) for v in got.values()))
    if not common:
        raise SystemExit("the allocators share no task sets")
    return ({n: (agg([v[t][0] for t in common]),
                 agg([v[t][1] for t in common]))
             for n, v in got.items()}, len(common))


def bar_style(name):
    """Family hue from figstyle. A bar cannot be dashed, so the criticality
    variants keep the convention by being hatched instead."""
    col, _, dash, _, _ = F.style(name)
    return col, ("///" if dash != "solid" else None)


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
    ap.add_argument("--stat", choices=("mean", "median"), default="mean",
                    help="how the per-instance penalties and times are "
                         "aggregated. The median is robust to the long tail "
                         "the mean is dominated by")
    ap.add_argument("--name", default=None,
                    help="default: onepass_quality, or onepass_quality_median")
    args = ap.parse_args()

    import paths
    # not `name`: the bar loop below binds that, and the last bar would win
    outname = args.name or ("onepass_quality" if args.stat == "mean"
                            else "onepass_quality_median")
    stats, n = gather(args.stat)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE)
    F.chrome(ax)

    xs, x = [], 0.0
    worst = max(stats[b][0] for b in BARS if b)
    for name in BARS:
        if name is None:
            x += 0.5
            continue
        v, ms = stats[name]
        col, hatch = bar_style(name)
        ax.bar(x, v, width=0.74, color=col, edgecolor=INK, lw=1.1,
               hatch=hatch, zorder=3)
        ax.text(x, v + worst * 0.022, f"{v:,.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK, zorder=4)
        # the time goes under the name rather than in a second panel. Against a
        # search at 1.5 s these differences vanish, so they only mean anything
        # here, next to the quality they bought
        xs.append((x, f"{TICK.get(name, name)}\n{ms:.2f} ms"))
        x += 1.0

    ax.set_xticks([p for p, _ in xs])
    ax.set_xticklabels([t for _, t in xs], fontsize=10)
    ax.set_xlim(xs[0][0] - 0.75, xs[-1][0] + 0.75)
    ax.set_ylim(0, worst * 1.12)
    ax.set_ylabel(f"{args.stat.capitalize()} penalty (lower is better)",
                  fontsize=10.5, color=MUTED)
    ax.tick_params(axis="x", length=0)

    fig.subplots_adjust(top=0.96, bottom=0.11, left=0.10, right=0.98)
    save(fig, paths.out(args.dir, outname))
    plt.close(fig)
    print(f"{n:,} task sets, {args.stat}")
    for b in BARS:
        if b:
            print(f"  {TICK[b]:<10}{stats[b][0]:>8.1f}{stats[b][1]:>10.2f} ms")


if __name__ == "__main__":
    main()
