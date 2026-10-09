"""Repair: place only the orphans, leave everything else running.

When a core fails its tasks are stranded. Moving any other task costs a stop
and a restart, so repair moves nothing except what the failure forced.

Each allocator applies its own rule to the shorter list. A packer starts from
the survivors' loads instead of from empty cores; the policy is seeded with the
survivors' cores and rolls out over the orphans only. So `repair` and `rebuild`
differ in what they may move and in nothing else.

    repair("bfd", model, reduced_platform, survivors)
    repair("rl",  model, reduced_platform, survivors)

`survivors` is {task_id: core_id} for every task that did not die with the
core. Anything missing from it is an orphan and gets placed.
"""

from . import heft
from .common import allocate as _pack

# every packer, and the rule it repairs with, which is the rule it builds with
PACKERS = {
    "bfd": dict(spread=False),
    "wfd": dict(spread=True),
    "bfd_crit": dict(spread=False, criticality=True),
    "wfd_crit": dict(spread=True, criticality=True),
    "heft": dict(rule="eft"),
    "heft_crit": dict(rule="eft", criticality=True),
}


def can_repair(name):
    """Does this allocator know how to repair?"""
    return name in PACKERS or name == "rl"


def repair(name, model, platform, survivors):
    """Place the orphans with `name`'s rule. Survivors do not move."""
    if name in PACKERS:
        kw = dict(PACKERS[name])
        if name == "heft":
            # heft orders by upward rank, which is a property of the task graph
            # and does not change when a core dies
            rank = heft.upward_rank(model)
            kw["order_key"] = lambda t: (-rank[t], t)
        return _pack(model, platform, pinned=survivors, **kw)

    if name == "rl":
        return _policy(name, model, platform, survivors)

    raise SystemExit(f"{name} has no repair mode. have: "
                     f"{', '.join(sorted(PACKERS) + ['rl'])}")


def _policy(name, model, platform, survivors):
    """The learned policy, seeded with where the survivors already are.

    `Episode` takes a seed of (task index -> core index) and treats those as
    already placed, so the rollout only chooses for the orphans. Same
    checkpoint, same greedy argmax, same features: nothing is repair specific.
    """
    import torch

    from rl.env import Instance
    from rl.rollout import rollout

    from .learned import _get

    policy, order, learn_order, device = _get(name)
    inst = Instance.from_model(model, platform, order=order)
    seed = {inst.task_index[t]: inst.core_index[c] for t, c in survivors.items()}
    with torch.no_grad():
        ep = rollout(policy, inst, greedy=True, device=device,
                     learn_order=learn_order, seed=seed)["episode"]
    return ep.allocation()
