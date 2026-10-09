"""Shared colour palette for the thesis figures.

Two encodings, kept on separate channels:

  hue       = cpu_type      A55 red, A78 yellow, Kryo_Gen6 blue
  lightness = core_class    os lightest, general, isolated darkest
                            (darker as the tail multiplier drops: 1.30, 1.21, 1.10)

Nine tints, all in a light band so labels stay dark ink throughout.
"""

# light -> dark, indexed by CLASS_STEP
CPU_RAMP = {
    "A55":       ["#ffe3e0", "#ffb3ab", "#f8837a"],
    "A78":       ["#fff0d4", "#ffd98a", "#f5bf52"],
    "Kryo_Gen6": ["#e2ecfd", "#b6d0f7", "#85aeea"],
}
UNKNOWN_RAMP = ["#f0efec", "#d5d4cf", "#b8b7b1"]
CLASS_STEP = {"os": 0, "general": 1, "isolated": 2}

LINK_COLOUR = {"usb": "#4a3aa7", "eth_1g": "#008300", "eth_10g": "#e87ba4"}
UNKNOWN_LINK = "#898781"

# criticality, derived from period. SOFT is a gradient: the slower the task, the
# darker the green.
HARD_FILL = "#C0504D"
FIRM_FILL = "#D9A441"
SOFT_GRADIENT = {1000: "#B7C84A", 2000: "#1F5728"}
SOFT_FALLBACK = "#6AA84F"
MISSED = "#111111"      # execution past the job's absolute deadline
GAP = "#6E6E6E"         # wake / dispatch / preempt


def crit_fill(period_us):
    if period_us == 100:
        return HARD_FILL
    if period_us == 500:
        return FIRM_FILL
    return SOFT_GRADIENT.get(period_us, SOFT_FALLBACK)

# chrome and ink
SURFACE = "#fcfcfb"
DEVICE_FILL = "#f7f7f7"
INK = "#0b0b0b"
INK_MUTED = "#898781"
HAIRLINE = "#c3c2b7"


def core_fill(cpu_type, core_class):
    return CPU_RAMP.get(cpu_type, UNKNOWN_RAMP)[CLASS_STEP[core_class]]


def link_colour(link_type):
    return LINK_COLOUR.get(link_type, UNKNOWN_LINK)


def _luminance(hex_colour):
    c = hex_colour.lstrip("#")
    chans = []
    for i in (0, 2, 4):
        v = int(c[i:i + 2], 16) / 255
        chans.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return 0.2126 * chans[0] + 0.7152 * chans[1] + 0.0722 * chans[2]


def _contrast(a, b):
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def ink_on(fill):
    """Ink or white label text, whichever has more contrast against the fill.

    Compares against the real INK value rather than pure black, which would
    pick the wrong side on mid-lightness fills.
    """
    return INK if _contrast(fill, INK) >= _contrast(fill, "#ffffff") else "#ffffff"
