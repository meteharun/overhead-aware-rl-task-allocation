"""Worst-fit decreasing, communication aware, and criticality ordered.

The same relationship to `wfd` that `bfd_crit` has to `bfd`. Having both means
the ordering can be judged on its own: if it helps one packer and hurts the
other, that is a result about the ordering rather than about either packer.
"""

from .common import allocate as _allocate


def allocate(model, platform):
    return _allocate(model, platform, spread=True, criticality=True)
