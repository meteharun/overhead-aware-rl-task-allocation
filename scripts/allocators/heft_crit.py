"""HEFT with the criticality ordering instead of the upward rank.

Every rule identical to `heft`, including the earliest finish time placement.
Only the task order changes.

HEFT's upward rank is a makespan heuristic: it ranks a task by how much work
sits behind it on the longest path, which says nothing about what missing it
costs. Where a soft task heads a long chain and a hard task sits on a short
one, the upward rank puts the soft task first and this variant does not.
"""

from .common import allocate as _allocate


def allocate(model, platform):
    return _allocate(model, platform, rule="eft", criticality=True)
