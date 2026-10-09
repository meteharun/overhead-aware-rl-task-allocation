#!/usr/bin/env python3
"""Per-instance penalty difference between the heuristic portfolio and RL.

    python scripts/fig/viz_portfolio.py

Writes out/writing/onepass-portfolio-vs-rl.png.

The aggregate tables say two things that sound contradictory. The portfolio beats
RL on more instances than it loses on, 3,001 against 2,350, and RL still has the
lower mean penalty by 30.9. Both are true, and neither number explains the other.

This sorts the 6,400 instances by the difference and plots it. The portfolio's
wins are many and mostly small. RL's wins are fewer and much larger. On a linear
axis the area under each lobe is that side's contribution to the mean, so the
figure is the mean difference taken apart: the portfolio earns 43.4 of it and RL
earns 74.3, leaving 30.9 to RL.

A win by 0.3 penalty is not a win. `--tie` sets the width of the band counted as
a draw. The floor for any honest band is 2.0, the most a single soft deadline
miss can cost, since internal weight is drawn from [1, 2] and the soft external
weight is 1, so two allocations closer than that are not one miss apart. The
default is 5.0, above that floor, and it is a judgement rather than a derivation.

What defends it is that nothing turns on where the line is drawn. The two mean
contributions move by less than 1.5 across every band from 0 to 20, because the
instances being reclassified are worth almost nothing. Only the count of wins
moves, from 46.9% against 36.7% at an exact tie to 30.7% against 35.3% here.
Since the count is the statistic the threshold does move, the figure reports the
band it used and `main` prints the whole sensitivity table.

Sign convention is penalty(Portfolio) minus penalty(RL), so positive is RL
ahead. Not an equal cost comparison: the portfolio is six constructive passes at
47.4 ms against one at 8.3 ms, which the caption has to say.
"""

import argparse
import json
import sys
from collections import defaultdict
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
from figstyle import HAIR, INK, MUTED, NEUTRAL, RL
from palette import SURFACE

ROOT = Path(__file__).resolve().parent.parent.parent
ALLOCS = ROOT / "data" / "allocations"

FIGSIZE = (11.0, 5.2)
DPI = 300
PAD = 18
OUTDIR = "writing"

GREEDY = ["bfd", "bfd_crit", "wfd", "wfd_crit", "heft", "heft_crit"]
LEARNED = "rl"

# the width of the band counted as a draw. The floor is 2.0, the most a single
# soft deadline miss can cost, below which two allocations are not one miss
# apart. This sits above it: see the module docstring and the sensitivity table
# `main` prints.
TIE_EPS = 5.0

BOX = dict(boxstyle="round,pad=0.35", facecolor=SURFACE, edgecolor=HAIR,
           linewidth=0.8, alpha=0.95)


def load():
    """penalty_hard for every method on every evaluation instance.

    The portfolio is the per-instance best of the six, which is how
    `analyse_allocations.add_portfolios` builds `best_greedy`. Rebuilt here
    rather than read from a file, because the difference is per instance and the
    saved tables carry only the aggregate.
    """
    pen = defaultdict(dict)
    for name in GREEDY + [LEARNED]:
        path = ALLOCS / f"{name}.jsonl"
        if not path.exists():
            raise SystemExit(f"no {path}. run scripts/run/run_allocators.py first")
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("set") != "eval":
                    continue
                pen[r["taskset_id"]][name] = r["summary"]["penalty_hard"]

    diffs = []
    for got in pen.values():
        if len(got) < len(GREEDY) + 1:
            continue                      # an allocator that did not cover it
        diffs.append(min(got[g] for g in GREEDY) - got[LEARNED])
    diffs.sort()
    return np.asarray(diffs, dtype=float)


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
    ap.add_argument("--name", default="onepass-portfolio-vs-rl")
    ap.add_argument("--tie", type=float, default=TIE_EPS, metavar="EPS",
                    help="differences at or below this count as a draw. "
                         "Default 2.0, the cost of one soft deadline miss")
    args = ap.parse_args()

    import paths
    d = load()
    n = len(d)
    xs = np.arange(1, n + 1)
    eps = args.tie

    win_p, win_r = d < -eps, d > eps
    n_port, n_rl = int(win_p.sum()), int(win_r.sum())
    n_tie = n - n_port - n_rl
    give = d[win_p].sum() / n         # negative, the portfolio's share
    take = d[win_r].sum() / n         # positive, RL's share
    net = d.mean()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE)
    F.chrome(ax)

    # the draws sit on zero and would otherwise be invisible, which hides a
    # quarter of the pool and makes every near miss look like a decision
    ax.axvspan(n_port, n_port + n_tie, color=HAIR, alpha=0.30, zorder=1)

    ax.fill_between(xs, d, 0, where=win_p, interpolate=True, color=NEUTRAL,
                    alpha=0.85, zorder=3, linewidth=0)
    ax.fill_between(xs, d, 0, where=win_r, interpolate=True, color=RL,
                    alpha=0.85, zorder=3, linewidth=0)
    ax.plot(xs, d, color=INK, lw=1.0, zorder=4)
    ax.axhline(0, color=INK, lw=1.0, zorder=5)

    lo, hi = d.min(), d.max()
    # a hair either side, or the first and last tick labels sit on the spine and
    # the 6,400 at the end is cut in half
    ax.set_xlim(-0.012 * n, 1.012 * n)
    ax.set_ylim(lo * 1.32, hi * 1.42)

    ax.set_xlabel("Evaluation instances, ordered by the difference", fontsize=10.5,
                  color=MUTED)
    ax.set_ylabel("penalty(Portfolio) − penalty(RL)", fontsize=10.5,
                  color=MUTED)
    ax.set_xticks([0, n // 4, n // 2, 3 * n // 4, n])
    ax.set_xticklabels([f"{v:,}" for v in
                        (0, n // 4, n // 2, 3 * n // 4, n)], fontsize=9.5)
    F.title(ax, "Per-instance penalty difference, heuristic portfolio against RL")

    # each lobe labelled inside the empty space above or below it, so neither
    # needs a leader line. The share of instances and the share of the mean are
    # the two numbers that disagree, so they sit together
    ax.text(0.20, 0.20, f"Portfolio lower\n{n_port:,} instances"
                        f"  ({100 * n_port / n:.1f}%)\n"
                        f"median margin {abs(np.median(d[win_p])):.1f}\n"
                        f"mean contribution {give:.1f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
            color=INK, zorder=6, bbox=BOX)
    ax.text(0.82, 0.80, f"RL lower\n{n_rl:,} instances"
                        f"  ({100 * n_rl / n:.1f}%)\n"
                        f"median margin {np.median(d[win_r]):.1f}\n"
                        f"mean contribution +{take:.1f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
            color=INK, zorder=6, bbox=BOX)
    ax.text((n_port + n_tie / 2) / n, 0.545,
            f"{n_tie:,} draws  ({100 * n_tie / n:.1f}%)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            color=MUTED, zorder=6)

    ax.text(0.015, 0.975, f"Mean difference  {net:+.1f}   in RL's favour",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color=RL, zorder=7, bbox=BOX)

    # which way is up. The sign convention is the one thing a reader can invert
    ax.annotate("RL lower penalty", xy=(0.995, 0.985), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color=MUTED)
    ax.annotate("Portfolio lower penalty", xy=(0.995, 0.015),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=9,
                color=MUTED)

    fig.subplots_adjust(top=0.91, bottom=0.115, left=0.085, right=0.99)
    save(fig, paths.out(args.dir, args.name))
    plt.close(fig)

    print(f"\n{n:,} instances, draw band |difference| <= {eps:g}")
    print(f"  portfolio lower  {n_port:>6,}  {100 * n_port / n:>5.1f}%   "
          f"median {abs(np.median(d[win_p])):>7.1f}   contributes {give:>7.1f}")
    print(f"  draw             {n_tie:>6,}  {100 * n_tie / n:>5.1f}%")
    print(f"  rl lower         {n_rl:>6,}  {100 * n_rl / n:>5.1f}%   "
          f"median {np.median(d[win_r]):>7.1f}   contributes {take:>+7.1f}")
    print(f"  net mean difference {net:+.2f}")
    print(f"  range {d.min():.1f} to {d.max():.1f}")

    # how much the verdict depends on where the draw band is drawn. If the
    # ordering flips inside this range the threshold is doing the arguing
    print(f"\n  {'eps':>6}{'portfolio':>12}{'draw':>10}{'rl':>10}{'net':>9}")
    for e in (0, 1, 2, 5, 10, 20):
        p, r = int((d < -e).sum()), int((d > e).sum())
        print(f"  {e:>6g}{100 * p / n:>11.1f}%{100 * (n - p - r) / n:>9.1f}%"
              f"{100 * r / n:>9.1f}%{net:>9.2f}")


if __name__ == "__main__":
    main()
