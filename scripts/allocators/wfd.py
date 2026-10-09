"""Worst-fit decreasing, communication aware.

Within the chosen device it takes the emptiest core, which keeps response times
down at the cost of filling the device sooner and spilling across the link
earlier. Everything else is `bfd`'s rule, in `common.allocate`.
"""

from .common import allocate as _allocate


def allocate(model, platform):
    return _allocate(model, platform, spread=True)
