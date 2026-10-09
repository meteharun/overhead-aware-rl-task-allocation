"""Best-fit decreasing, communication aware.

Within the chosen device it takes the fullest core with room, which packs the
main device tightly and so keeps tasks off the link. Device choice, the os core
reservation and the crossing cost are all in `common.allocate`.
"""

from .common import allocate as _allocate


def allocate(model, platform):
    return _allocate(model, platform, spread=False)
