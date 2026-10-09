"""Best-fit decreasing, communication aware, and criticality ordered.

Identical to `bfd` in every rule. The only difference is which task it takes
next: among those whose predecessors are all placed, the one that would cost
the most to miss, under the same weighting the objective uses.

Plain bfd walks the graph in whatever topological order falls out, so a soft
task early in a chain can take a good core before a hard task later in it. This
variant does not, while keeping the dependency ordering the communication rule
needs.
"""

from .common import allocate as _allocate


def allocate(model, platform):
    return _allocate(model, platform, spread=False, criticality=True)
