"""Draw a platform: cores inside device boxes, links between them.

    python scripts/visualization/viz_platform.py 3         data/platforms/platform_3.json
    python scripts/visualization/viz_platform.py 3 7 10    several at once
    python scripts/visualization/viz_platform.py           every platform there is

Writes out/platforms/<name>.png
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import paths
from palette import (DEVICE_FILL, HAIRLINE, INK, INK_MUTED, SURFACE,
                     core_fill, ink_on, link_colour)

CORE_W, CORE_H = 2.4, 0.8
PAD = 0.3
TITLE_H = 0.9
ROW_PITCH = CORE_H + 0.18
DEV_GAP = 1.8
CLASSES = ["os", "general", "isolated"]

ROOT = Path(__file__).resolve().parent.parent.parent


def draw_legend(ax, cpu_types, link_types, link_tbl, y):
    x = 0.0
    for cpu in cpu_types:
        ax.text(x, y + 0.5, cpu, fontsize=8, weight="bold", color=INK)
        for k, cls in enumerate(CLASSES):
            ax.add_patch(FancyBboxPatch((x + k * 0.64, y), 0.56, 0.32,
                                        boxstyle="round,pad=0.02",
                                        fc=core_fill(cpu, cls), ec=HAIRLINE, lw=0.8))
        for k, cls in enumerate(CLASSES):
            ax.text(x + k * 0.64 + 0.28, y - 0.28, cls, fontsize=6,
                    ha="center", va="top", color=INK_MUTED)
        x += 3.1

    for lt in link_types:
        ax.plot([x, x + 0.9], [y + 0.16, y + 0.16], lw=2.4, color=link_colour(lt))
        t = link_tbl[lt]
        ax.text(x + 1.05, y + 0.16, f"{lt}   {t['delay_us']} us / {t['cpu_us']} us cpu",
                fontsize=7, va="center", color=INK)
        x += 4.6


def main(path):
    plat = json.loads(Path(path).read_text())
    hw = json.loads((ROOT / "data" / "catalogues" / "hardware_catalogue.json").read_text())
    link_tbl = {lt["link_type"]: lt for lt in hw["link_types"]}
    tail = hw["core_class_tail"]

    by_dev = defaultdict(list)
    for c in plat["cores"]:
        by_dev[c["device_id"]].append(c)
    order = [d["device_id"] for d in plat["devices"]]

    fig, ax = plt.subplots(figsize=(2.5 + 4.4 * len(order), 9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    tops, x = {}, 0.0
    for dev in order:
        cores = by_dev[dev]
        rows = (len(cores) + 1) // 2
        w = 2 * CORE_W + 3 * PAD
        h = TITLE_H + rows * ROW_PITCH + PAD

        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.12",
                                    fc=DEVICE_FILL, ec=INK, lw=1.6))
        summary = ", ".join(f"{n}x {t}" for t, n in
                            Counter(c["cpu_type"] for c in cores).items())
        ax.text(x + w / 2, h - TITLE_H / 2, f"{dev}\n{summary}",
                ha="center", va="center", fontsize=11, weight="bold", color=INK)

        for j, c in enumerate(cores):
            r, col = divmod(j, 2)
            cx = x + PAD + col * (CORE_W + PAD)
            cy = h - TITLE_H - (r + 1) * ROW_PITCH
            fill = core_fill(c["cpu_type"], c["core_class"])
            ax.add_patch(FancyBboxPatch((cx, cy), CORE_W, CORE_H,
                                        boxstyle="round,pad=0.04",
                                        fc=fill, ec=HAIRLINE, lw=0.8))
            ax.text(cx + CORE_W / 2, cy + CORE_H / 2,
                    f"{c['core_id']}\n{c['cpu_type']}  {c['core_class']}  "
                    f"tail {tail[c['core_class']]}",
                    ha="center", va="center", fontsize=6.5, color=ink_on(fill))

        tops[dev] = (x + w / 2, h)
        x += w + DEV_GAP

    max_top = max(t[1] for t in tops.values()) if tops else 1.0

    arc_top = max_top
    for lk in plat["links"]:
        (xa, ha), (xb, hb) = tops[lk["dev_a"]], tops[lk["dev_b"]]
        span = abs(order.index(lk["dev_a"]) - order.index(lk["dev_b"]))
        colour = link_colour(lk["link_type"])
        rad = -0.16 * span
        ax.add_patch(FancyArrowPatch((xa, ha), (xb, hb), arrowstyle="-",
                                     connectionstyle=f"arc3,rad={rad}",
                                     lw=2.4, color=colour))
        # apex of the quadratic bezier matplotlib builds for arc3, so the label
        # sits on its own arc instead of floating between two of them
        dx, dy = xb - xa, hb - ha
        ax_, ay = (xa + xb) / 2 + 0.5 * rad * dy, (ha + hb) / 2 - 0.5 * rad * dx
        arc_top = max(arc_top, ay)
        t = link_tbl[lk["link_type"]]
        ax.text(ax_, ay, f"{lk['link_type']}\ndelay {t['delay_us']} us\ncpu {t['cpu_us']} us",
                ha="center", va="center", fontsize=8, color=colour,
                bbox=dict(boxstyle="round", fc=SURFACE, ec=colour))

    cpus = sorted({c["cpu_type"] for c in plat["cores"]})
    links = sorted({lk["link_type"] for lk in plat["links"]})
    draw_legend(ax, cpus, links, link_tbl, y=-1.9)

    ax.set_xlim(-1, max(x, 3.1 * len(cpus) + 4.6 * len(links)))
    ax.set_ylim(-2.7, arc_top + 1.4)
    ax.set_aspect("equal")
    ax.set_title(f"{plat['platform_id']}  |  {len(order)} devices, "
                 f"{len(plat['cores'])} cores, {len(plat['links'])} links", color=INK)
    ax.axis("off")
    fig.tight_layout()

    out = paths.out("platforms", Path(path).stem)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    # drawing ten of these in one process otherwise piles up open figures until
    # matplotlib starts complaining
    plt.close(fig)


def every():
    """All platforms on disk, in numeric order rather than platform_10 first."""
    files = paths.DATA.glob("platforms/platform_*.json")
    return sorted(files, key=lambda f: int(f.stem.split("_")[1]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("platforms", nargs="*",
                    help="platform numbers. Omit for all of them")
    args = ap.parse_args()

    targets = ([paths.platform(p) for p in args.platforms] if args.platforms
               else every())
    if not targets:
        raise SystemExit("no platforms found. run scripts/gen_platform.py first")
    for f in targets:
        main(f)
