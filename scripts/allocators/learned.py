"""The trained policy, wrapped as an ordinary allocator.

Registered under the name `rl`, so `run_allocators`, `analyse_allocations` and
`viz_alloc` treat it exactly like bfd or wfd, and `sa_rl` falls out of the same
registry rule that produced sa_bfd and sa_wfd.

Placement order is fixed: criticality on a single device, topological
otherwise, and the policy only chooses the core. How a checkpoint was trained
is stored in the checkpoint, so it cannot be run the wrong way.

The checkpoint is loaded once and kept, since inference is meant to be a single
forward pass and reloading per task set would hide that. torch is imported
inside the functions, so scripts that never ask for a learned allocation do not
pay for it.

Inference is the greedy rollout, argmax at every step and no sampling, because
`viz_alloc` recomputes the allocation rather than reading the saved one and has
to land on the same answer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"

CHECKPOINTS = {"rl": ROOT / "data" / "models" / "rl.pt"}

CHECKPOINT = CHECKPOINTS["rl"]

_loaded = {}


def available(name="rl"):
    return name in CHECKPOINTS and CHECKPOINTS[name].exists()


def trained():
    """The names whose checkpoint is on disk, in registration order."""
    return [n for n in CHECKPOINTS if available(n)]


def _get(name):
    """Load one checkpoint once. Returns (policy, order, learn_order, device)."""
    if name not in _loaded:
        path = CHECKPOINTS[name]
        if not path.exists():
            raise SystemExit(
                f"no trained policy at {path}. "
                "run: python scripts/rl/train.py")
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        import torch
        from rl.policy import load
        torch.set_num_threads(1)
        policy, meta = load(path, device="cpu")
        # how it was trained is how it must be run, so both come from the file
        _loaded[name] = (policy, meta.get("order", "topo"),
                         bool(meta.get("learn_order", False)), "cpu")
    return _loaded[name]


def preload(names=None):
    """Read the checkpoints now rather than inside the first allocation.

    Otherwise importing torch and reading the file land inside the first timed
    call, and the first row of the timing table measures loading, not
    inference.
    """
    for n in (names or trained()):
        _get(n)


def make(name):
    """An allocate(model, platform) bound to one checkpoint."""

    def allocate(model, platform):
        import torch
        from rl.env import Instance
        from rl.rollout import rollout

        policy, order, learn_order, device = _get(name)

        # Instance normally builds its own Model from json. Here it must reuse
        # the caller's, so the evaluation afterwards sees the same state.
        inst = Instance.from_model(model, platform, order=order)

        with torch.no_grad():
            return rollout(policy, inst, greedy=True, device=device,
                           learn_order=learn_order)["episode"].allocation()

    allocate.__name__ = name
    return allocate


allocate = make("rl")
