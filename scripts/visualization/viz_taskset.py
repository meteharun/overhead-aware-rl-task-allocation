"""Draw a task set: nodes are tasks, edges are data flow.

    python scripts/visualization/viz_taskset.py 10_3_3_3   by id, out of the store
    python scripts/visualization/viz_taskset.py one.json   or a task set dumped to a file

Writes out/tasksets/<name>.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import paths
import tasksets
from palette import (FIRM_FILL, HAIRLINE, HARD_FILL, SOFT_GRADIENT, SURFACE,
                     crit_fill, ink_on)

NODE_SIZE = 1500
MAX_COLS = 14      # layers per band before the graph wraps to a new row


def show(v):
    return "?" if v is None else v


def main(path):
    inst = (json.loads(Path(path).read_text()) if str(path).endswith(".json")
            else tasksets.load(path))

    g = nx.DiGraph()
    for t in inst["tasks"]:
        g.add_node(t["task_id"], **t)
    for e in inst["edges"]:
        g.add_edge(e["src"], e["dst"])

    # One column per topological layer, left to right. A long chain would be a
    # single strip thousands of pixels wide, so it wraps onto further bands.
    # Tasks with no edge at all would otherwise land in generation 0 and make
    # the first column enormous, so they get their own band at the bottom.
    loose = sorted(n for n in g.nodes if g.degree(n) == 0)
    linked = g.subgraph([n for n in g.nodes if g.degree(n) > 0])

    pos = {}
    layers = list(nx.topological_generations(linked)) if linked.number_of_nodes() else []
    bands = [layers[i:i + MAX_COLS] for i in range(0, len(layers), MAX_COLS)]

    # each band is only as tall as its own deepest layer, so a band of single
    # nodes does not reserve the height of the widest one in the whole graph
    y_off = 0.0
    for band in bands:
        for c, layer in enumerate(band):
            for row, n in enumerate(sorted(layer)):
                pos[n] = (c * 1.0, -(row + y_off))
        y_off += max(len(l) for l in band) + 1.4

    loose_rows = 0
    if loose:
        y_off += 0.6
        for i, n in enumerate(loose):
            r, c = divmod(i, MAX_COLS)
            pos[n] = (c * 1.0, -(r + y_off))
        loose_rows = (len(loose) + MAX_COLS - 1) // MAX_COLS
        ax_loose_y = -(y_off - 0.7)
        y_off += loose_rows
    total_h = y_off

    # id, then deadline over period, then weight. Criticality is the colour, so
    # spelling it out again only made the boxes bigger.
    labels, colors = {}, []
    for n in g.nodes:
        t = g.nodes[n]
        labels[n] = (f"{n}\n{show(t['deadline_us'])}/{t['period_us']}\n"
                     f"w={show(t['internal_weight'])}")
        colors.append(crit_fill(t["period_us"]))

    fig, ax = plt.subplots(figsize=(max(11, 1.15 * min(len(layers), MAX_COLS) + 6),
                                    max(5, 0.62 * total_h + 2.0)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # every edge is sampled, so there is nothing to distinguish
    nx.draw_networkx_edges(g, pos, ax=ax, node_size=NODE_SIZE, arrowsize=9,
                           edge_color="#888888", width=0.8,
                           connectionstyle="arc3,rad=0.08")

    nx.draw_networkx_nodes(g, pos, node_color=colors, node_size=NODE_SIZE,
                           node_shape="s", edgecolors=HAIRLINE, linewidths=0.8, ax=ax)
    for n, (x, y) in pos.items():
        ax.text(x, y, labels[n], ha="center", va="center", fontsize=5.5,
                color=ink_on(crit_fill(g.nodes[n]["period_us"])), zorder=3)

    if loose:
        ax.text(-0.6, ax_loose_y, f"independent, {len(loose)} task(s), no edges",
                ha="left", va="center", fontsize=8, style="italic",
                color="#52514e")

    lines = []
    for b in inst["budgets"]:
        lines.append(f"{b['src']} -> {b['dst']}   budget {show(b['budget_us'])}")
        for p in nx.all_simple_paths(g, b["src"], b["dst"]):
            floor = sum(g.nodes[n]["period_us"] for n in p[1:])
            lines.append(f"  floor {floor}, {len(p) - 1} hops")
            for i in range(0, len(p), 8):        # a long path has to wrap
                lines.append("    " + " > ".join(p[i:i + 8]))
    budgets = "\n".join(lines)
    ax.text(1.005, 0.99, f"end-to-end budgets\n{budgets}",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", fc=SURFACE, ec=HAIRLINE))

    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, mec=HAIRLINE,
                          mfc=c, label=lab)
               for c, lab in ((HARD_FILL, "HARD (100 us)"),
                              (FIRM_FILL, "FIRM (500 us)"),
                              (SOFT_GRADIENT[1000], "SOFT (1 ms)"),
                              (SOFT_GRADIENT[2000], "SOFT (2 ms)"))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize=8, frameon=False)

    ax.set_title(f"{inst['taskset_id']}  |  {len(g.nodes)} tasks, {len(g.edges)} edges"
                 f",  {len(loose)} independent"
                 f"   node label: id / deadline over period / weight")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.72, top=0.94, bottom=0.08)

    out = paths.out("tasksets", inst["taskset_id"])
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("taskset", help="task set id, with or without the taskset_ prefix")
    main(ap.parse_args().taskset)
