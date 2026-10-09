"""Allocator registry. Every allocator is allocate(model, platform) -> {task_id: core_id}.

Two kinds live here.

  base      one pass, constructive, no search
  annealed  a base allocation handed to simulated annealing, registered
            automatically as sa_<base> with an identical search budget, so the
            starting point is the only difference between them
"""

from . import bfd, bfd_crit, exact, heft, heft_crit, learned, sa, wfd, wfd_crit
from .repair import can_repair, repair  # noqa: F401  re-exported for repair.py

BASE = {
    "bfd": bfd.allocate,
    "wfd": wfd.allocate,
    # the same two packers, taking the most critical ready task rather than the
    # first, so a win over them is not a win over a baseline that ignores
    # task weights
    "bfd_crit": bfd_crit.allocate,
    "wfd_crit": wfd_crit.allocate,
    # bfd and wfd rank cores by how full they are, heft by how long this task
    # would take on them, the standard list heuristic for a dependent graph on
    # unequal processors
    "heft": heft.allocate,
    "heft_crit": heft_crit.allocate,
}

# The learned policy appears only once its checkpoint is on disk, so scripts
# that run "all allocators" still work before training has happened.
for _n in learned.trained():
    BASE[_n] = learned.make(_n)

# The optimality reference, not a heuristic and not comparable to one on time.
# It runs under a per instance time limit and reports whether it proved
# optimality or ran out.
BASE["exact"] = exact.make()

# annealing a proved optimum cannot improve it, and annealing a timed out one
# would report a search result under a name claiming exactness
NO_ANNEAL = {"exact"}

ANNEALED = {f"sa_{n}": sa.wrap(f, f"sa_{n}")
            for n, f in BASE.items() if n not in NO_ANNEAL}

ALLOCATORS = {**BASE, **ANNEALED}

REFERENCE = "bfd"      # what budgets and reward normalisation are measured against

# Dataset quality is judged on these two and no others: the grid was tuned so
# that bfd and wfd disagree on it, and admitting a search or a learned policy
# would move the verdict on the dataset without the dataset changing.
QUALITY_JUDGES = ("bfd", "wfd")


def configure_sa(iters):
    """Rebuild every annealed allocator with a different search budget.

    The thesis compares starting points at 4,000 evaluations. Asking whether an
    instance is solvable at all wants a much larger budget from the same
    search, so it is a setting rather than a separate script. The checkpoint
    ladder extends with the budget and the short run's points stay put.
    """
    global ANNEALED, ALLOCATORS
    ANNEALED = {f"sa_{n}": sa.wrap(f, f"sa_{n}", iters=iters)
                for n, f in BASE.items() if n not in NO_ANNEAL}
    ALLOCATORS = {**BASE, **ANNEALED}
    return iters


def configure_exact(**kw):
    """Rebuild the exact allocator with different settings.

    Its time limit is the one knob that moves between runs: seconds per
    instance on a pilot, minutes on the small platforms where the search can
    finish. Everything goes through the registry, so rebinding here is enough.
    """
    ALLOCATORS["exact"] = BASE["exact"] = exact.make(**kw)
    return ALLOCATORS["exact"]


def get(name):
    if name not in ALLOCATORS:
        raise SystemExit(f"unknown allocator {name!r}. have: {', '.join(names())}")
    return ALLOCATORS[name]


def names():
    return sorted(ALLOCATORS)


def base_of(name):
    """sa_bfd -> bfd, bfd -> None."""
    return name[3:] if name.startswith("sa_") else None
