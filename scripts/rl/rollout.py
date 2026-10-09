"""Run one episode. Shared by training, the allocator wrapper and the repair test.

Two modes. Under the fixed order the instance decides which task comes next and
the policy only chooses a core, which is directly comparable to `bfd` and `wfd`
since they walk the same order. Under the learned order the policy also chooses
which ready task to place, the same freedom `bfd_crit` has, as two decisions per
step rather than one joint decision over every (task, core) pair.
"""

import numpy as np
import torch

from .env import Episode


def instance_tensors(inst, device):
    """The per instance constants, moved to torch once."""
    return (torch.as_tensor(inst.task_feat, device=device),
            torch.as_tensor(inst.adj_in, device=device),
            torch.as_tensor(inst.adj_out, device=device),
            torch.as_tensor(inst.core_feat, device=device))


def _summary(pair):
    """Collapse a task's [M, F] pair features into [3F]: best, mean, worst core.

    Min and max are taken feature by feature rather than by picking one core,
    so the emptiest core available and the cheapest execution available can
    come from different cores.
    """
    return np.concatenate([pair.min(0), pair.mean(0), pair.max(0)])


def rollout(policy, inst, greedy=False, device="cpu", temperature=1.0,
            learn_order=False, seed=None):
    """Place every task. Returns the episode, its score, and the training terms.

    greedy   take the argmax. This is both the inference path and the baseline
             the sampled episode is scored against during training.
    seed     placements that are already fixed. Used by the repair test, where
             everything that survived a core failure stays put.
    """
    tf, ai, ao, cf = instance_tensors(inst, device)
    h, g = policy.encode(tf, ai, ao, cf)

    ep = Episode(inst, seed=seed)
    logps, ents = [], []

    def choose(z):
        """Sample or take the argmax, recording what training needs."""
        if greedy:
            return int(torch.argmax(z))
        dist = torch.distributions.Categorical(logits=z / temperature)
        a = dist.sample()
        logps.append(dist.log_prob(a))
        ents.append(dist.entropy())
        return int(a)

    while not ep.done():
        if learn_order:
            cand = ep.ready()
            pairs = [ep.features(n) for n in cand]
            if len(cand) > 1:
                summaries = torch.as_tensor(
                    np.stack([_summary(p) for p in pairs]), device=device)
                which = choose(policy.pick_logits(h[cand], summaries))
            else:
                which = 0
            n, pair = cand[which], pairs[which]
        else:
            n = ep.next_task()
            pair = ep.features(n)

        z = policy.logits(h[n], g, torch.as_tensor(pair, device=device))
        ep.place(n, choose(z))

    out = {"episode": ep, "score": ep.score()}
    if not greedy:
        # one log probability for the whole episode: REINFORCE credits every
        # decision with the same terminal advantage
        out["logp"] = torch.stack(logps).sum()
        out["entropy"] = torch.stack(ents).mean()
    return out
