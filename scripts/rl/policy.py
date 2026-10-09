"""The policy network. Small on purpose.

Three pieces:

  task encoder   a graph network over the task DAG, message passing in both
                 directions. Runs once per episode, since the task graph does
                 not change while an allocation is being built.

  core encoder   an MLP over the static core features, also once per episode.
                 Everything about a core that changes as tasks are placed is in
                 the pair features instead.

  scoring head   an MLP over [task embedding, core embedding, pair features],
                 one row per core. Applied per pair rather than to a fixed size
                 output layer, so the number of cores can be anything, which is
                 what makes an unseen platform work.

No attention and no bilinear product: at this size the concatenation head is as
good and there is one less thing to tune.
"""

import torch
import torch.nn as nn

from .env import (CORE_FEATURES, N_CORE_FEATURES, N_PAIR_FEATURES,
                  N_TASK_FEATURES, PAIR_FEATURES, TASK_FEATURES)


class GraphLayer(nn.Module):
    """h <- LayerNorm(h + MLP([h, mean of predecessors, mean of successors]))."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, a_in, a_out):
        return self.norm(h + self.mlp(torch.cat([h, a_in @ h, a_out @ h], dim=-1)))


class Policy(nn.Module):
    def __init__(self, dim=64, layers=3, hidden=128):
        super().__init__()
        self.dim = dim
        self.task_in = nn.Linear(N_TASK_FEATURES, dim)
        self.graph = nn.ModuleList([GraphLayer(dim) for _ in range(layers)])
        self.core_in = nn.Sequential(
            nn.Linear(N_CORE_FEATURES, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head = nn.Sequential(
            nn.Linear(2 * dim + N_PAIR_FEATURES, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

        # Second head, used only when the placement order is learned as well.
        # It scores one ready task at a time from its embedding plus a summary
        # of what the platform looks like to it. Two heads rather than one over
        # every (task, core) pair costs R + M rows per step instead of R x M.
        self.pick = nn.Sequential(
            nn.Linear(dim + N_PAIR_FEATURES * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def encode(self, task_feat, adj_in, adj_out, core_feat):
        """Everything that is fixed for the whole episode."""
        h = self.task_in(task_feat)
        for layer in self.graph:
            h = layer(h, adj_in, adj_out)
        return h, self.core_in(core_feat)

    def logits(self, h_task, core_emb, pair_feat):
        """One logit per core, for the single task being placed right now."""
        x = torch.cat(
            [h_task.unsqueeze(0).expand(core_emb.shape[0], -1), core_emb, pair_feat],
            dim=-1)
        return self.head(x).squeeze(-1)

    def pick_logits(self, h_tasks, pair_summaries):
        """One logit per ready task: which one to place next.

        `pair_summaries` is [R, 3 * N_PAIR_FEATURES], the best, mean and worst
        core for each candidate, feature by feature. That is what separates a
        task with a good home available from one with nowhere left to go.
        """
        return self.pick(torch.cat([h_tasks, pair_summaries], dim=-1)).squeeze(-1)


def save(policy, path, meta=None):
    torch.save({"state_dict": policy.state_dict(),
                "dim": policy.dim,
                "layers": len(policy.graph),
                "hidden": policy.head[0].out_features,
                "meta": meta or {}}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck.get("meta", {})

    # A checkpoint is only meaningful alongside the features it was trained on.
    # Without this check a changed feature set either crashes with an unhelpful
    # shape error or loads and quietly means something else.
    for key, current in (("task_features", TASK_FEATURES),
                         ("core_features", CORE_FEATURES),
                         ("pair_features", PAIR_FEATURES)):
        saved = meta.get(key)
        if saved is not None and list(saved) != list(current):
            added = [f for f in current if f not in saved]
            gone = [f for f in saved if f not in current]
            raise SystemExit(
                f"{path} was trained on a different feature set ({key}: "
                f"{len(saved)} then, {len(current)} now"
                + (f", added {added}" if added else "")
                + (f", removed {gone}" if gone else "")
                + "). Retrain: python scripts/rl/train.py")

    p = Policy(dim=ck["dim"], layers=ck["layers"], hidden=ck["hidden"])

    # A checkpoint trained before the ordering head existed has no weights for
    # it. That is fine as long as the order was fixed, because the head is then
    # never called; it is fatal if the order was learned.
    missing, unexpected = p.load_state_dict(ck["state_dict"], strict=False)
    if unexpected:
        raise SystemExit(f"{path} has weights this network has no place for: "
                         f"{sorted(unexpected)}. Retrain.")
    if missing:
        if meta.get("learn_order") or any(not k.startswith("pick.") for k in missing):
            raise SystemExit(f"{path} is missing {sorted(missing)}. Retrain: "
                             "python scripts/rl/train.py")

    p.to(device).eval()
    return p, meta
