"""One look and one colour meaning, shared by every result figure.

  blue     the learned policy, the only saturated hue and the only heavy line
  amber    best fit
  violet   worst fit
  teal     heft, earliest finish time
  green    simulated annealing, where it appears as a method not as an axis

Within a family the plain variant is solid and the criticality-ordered variant
is the same hue one step darker and dashed. These four hues also separate under
red-green colour blindness.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from palette import (FIRM_FILL, HAIRLINE, HARD_FILL, INK, INK_MUTED,
                     SOFT_FALLBACK, SURFACE)

HAIR, MUTED = HAIRLINE, INK_MUTED

RL = "#1B4F9C"          # the policy
RL_PALE = "#7FA6D9"     # the policy, constrained
BFD = "#E3B278"
BFD_DARK = "#C08234"
BFD_PALE = "#F1D9BC"
WFD = "#B8A6D6"
WFD_DARK = "#8264B0"
WFD_PALE = "#DCD3EB"
HEFT = "#2E8B8B"
HEFT_DARK = "#1F6363"
HEFT_PALE = "#97C5C5"
ANNEAL = "#4E9A6A"
NEUTRAL = "#9A9A94"
NEUTRAL_PALE = "#CCCCC8"

GOOD = "#2E7D32"
BAD_X = "#C62828"       # a deadline, and anything that failed to meet one
DASH = (0, (5, 2))

# the two halves of penalty_hard, as two tones of one slate. Not the
# criticality reds, which mean HARD specifically.
MISS_PEN = "#3F4A54"
CHAIN_PEN = "#8FAFC4"

# criticality, straight from palette so a HARD task is the same red here as in
# every schedule and platform figure
CRIT = {"hard": HARD_FILL, "firm": FIRM_FILL, "soft": SOFT_FALLBACK}

# name -> colour, line width, dash, marker size, z. Anything not listed falls
# back to NEUTRAL, so a new allocator shows up grey rather than stealing a hue.
STYLE = {
    "rl":          (RL, 3.0, "solid", 7.0, 6),
    "bfd":         (BFD, 1.9, "solid", 4.5, 3),
    "bfd_crit":    (BFD_DARK, 1.9, DASH, 4.5, 3),
    "wfd":         (WFD, 1.9, "solid", 4.5, 3),
    "wfd_crit":    (WFD_DARK, 1.9, DASH, 4.5, 3),
    "heft":        (HEFT, 1.9, "solid", 4.5, 3),
    "heft_crit":   (HEFT_DARK, 1.9, DASH, 4.5, 3),
    "anneal":      (ANNEAL, 1.9, "solid", 4.5, 3),
    "exact":       (INK, 2.2, (0, (1, 2)), 5.0, 7),
}
FALLBACK = (NEUTRAL, 1.9, "solid", 4.5, 3)

# the washed out version of each hue, for an allocator working under a
# constraint. Repair reads as the same allocator, not a different one.
PALE = {"rl": RL_PALE, "bfd": BFD_PALE, "wfd": WFD_PALE, "heft": HEFT_PALE}

LABEL = {
    "rl": "RL",
    "bfd": "best fit, topological order",
    "bfd_crit": "best fit, criticality order",
    "wfd": "worst fit, topological order",
    "wfd_crit": "worst fit, criticality order",
    "heft": "earliest finish time",
    "heft_crit": "earliest finish time, criticality order",
    "anneal": "search for a new plan from scratch",
    "exact": "optimal",
}

# said the way someone would say it out loud, not the way the code names it
FREED = {"0": "place only the stranded tasks",
         "rebuild": "replan everything"}

# for axis ticks and dense legends, where LABEL is too long
SHORT = {
    "bfd": "BFD", "bfd_crit": "BFD-C",
    "wfd": "WFD", "wfd_crit": "WFD-C",
    "heft": "HEFT", "heft_crit": "HEFT-C",
    "rl": "RL",
    "best_greedy": "Portfolio", "best_greedy_sa": "Portfolio+SA",
    "oracle": "Oracle", "exact": "Exact", "anneal": "SA",
}


def style(name):
    """The look for one series.

    Repair results are named `allocator:freed` and resolve to their own
    allocator's hue, washed out in proportion to how little they may move.
    Annealed allocators wear the colour of the allocation they started from.
    """
    if name in STYLE:
        return STYLE[name]
    if ":" in name:
        base, freed = name.split(":", 1)
        col, lw, _, ms, z = STYLE.get(base, FALLBACK)
        if freed == "rebuild":
            return (col, lw, "solid", ms, z)
        if freed == "0":
            return (PALE.get(base, NEUTRAL_PALE), max(lw - 0.8, 1.5), DASH,
                    max(ms - 1.0, 3.5), z)
        return (col, max(lw - 0.6, 1.6), DASH, max(ms - 0.5, 4.0), z)
    if name.startswith("sa_"):
        return style(name[3:])
    return FALLBACK


def short(name):
    """The abbreviation.

    `sa_bfd` keeps its prefix and reads SA_BFD. A chart whose x axis is the
    search budget passes the stripped name instead.
    """
    if name in SHORT:
        return SHORT[name]
    if name.startswith("sa_"):
        base = name[3:]
        return "SA_" + SHORT.get(base, base)
    return SHORT.get(name, name)


def label(name):
    if name in LABEL:
        return LABEL[name]
    if ":" in name:
        base, freed = name.split(":", 1)
        what = FREED.get(freed, f"free {freed} running tasks")
        return f"{LABEL.get(base, base)}: {what}"
    return name


def colour(name):
    return style(name)[0]


def chrome(*axes):
    """The frame every panel wears. Grid behind, no box, muted ticks."""
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(color=HAIR, linewidth=0.6, alpha=0.7)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(HAIR)
        a.tick_params(colors=MUTED, labelsize=8.5)


def title(ax, text):
    ax.set_title(text, fontsize=11.5, color=INK, loc="left", fontweight="bold")


def suptitle(fig, text):
    fig.suptitle(text, fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)


def legend(fig, names, bars=False):
    """One legend for the figure, at the bottom, subject first."""
    handles = []
    for n in names:
        col, lw, dash, ms, _ = style(n)
        if bars:
            handles.append(Patch(facecolor=col, edgecolor=INK, lw=1.0,
                                 label=label(n)))
        else:
            handles.append(Line2D([], [], color=col, lw=lw, linestyle=dash,
                                  marker="o", markersize=ms,
                                  markeredgecolor=SURFACE, markeredgewidth=1.2,
                                  label=label(n)))
    fig.legend(handles=handles, loc="lower center", ncol=len(names),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
