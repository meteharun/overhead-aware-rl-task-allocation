"""Train the placement policy.

    python scripts/rl/train.py                              defaults
    python scripts/rl/train.py --steps 200 --batch 8        a smoke run, minutes
    python scripts/rl/train.py --warm-from sa_bfd           better labels, if you have them
    python scripts/rl/train.py --device cuda

Writes data/models/rl.pt (best validation) and data/models/rl_last.pt. The
allocator registry picks up data/models/rl.pt on its own, so once this finishes
`rl` and `sa_rl` appear in run_allocators, analyse_allocations and viz_alloc.

REINFORCE with a greedy rollout baseline. Each instance is run twice, once
sampling and once taking the argmax, and the sampled episode is rewarded for
beating its own greedy self. That costs one extra rollout and removes the need
to fit a value function. The advantage is normalised per instance, since
penalty ranges from zero to several hundred depending on the platform and a raw
difference would let a few hard instances own the gradient.

The reward is `penalty.total`, the thesis objective. Validation also reports
`penalty_hard`, which is what the allocator comparison uses.

Optional warm start: before the RL loop the policy can be trained by cross
entropy to reproduce an allocator whose allocations are already in
data/allocations/<name>.jsonl. A cold random policy on a forty task, twelve
core problem is slow and often settles for no better than the greedy packers.
`--warm-from sa_bfd` is a better teacher than `--warm-from bfd`.
"""

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import tasksets  # noqa: E402
from allocators import get  # noqa: E402
from progress import hms  # noqa: E402
from rl import env as envmod  # noqa: E402
from rl.env import Episode, load_instance  # noqa: E402
from rl.policy import Policy, save  # noqa: E402
from rl.rollout import instance_tensors, rollout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS = tasksets.DIR
ALLOCS = ROOT / "data" / "allocations"
MODELS = ROOT / "data" / "models"


class Cache:
    """Bounded instance cache. Building an Instance costs more than running an
    episode on it, and the whole training set does not fit in memory."""

    def __init__(self, cap, order):
        self.cap, self.order = cap, order
        self.d = OrderedDict()

    def get(self, tsid):
        if tsid in self.d:
            self.d.move_to_end(tsid)
            return self.d[tsid]
        inst = load_instance(tsid, order=self.order)
        self.d[tsid] = inst
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return inst


def read_labels(name):
    """Allocations already on disk, keyed by task set id."""
    f = ALLOCS / f"{name}.jsonl"
    if not f.exists():
        raise SystemExit(f"no {f}. run scripts/run/run_allocators.py --only {name} first")
    out = {}
    with f.open() as fh:
        for line in fh:
            r = json.loads(line)
            out[r["taskset_id"]] = r["allocation"]
    return out


def warm_step(policy, insts, labels, device):
    """Cross entropy against a stored allocation, with teacher forcing.

    Also returns top-1 agreement with the teacher, which is easier to read than
    raw cross entropy: its floor depends on the core count, and equivalent
    cores make the label partly arbitrary, so perfect agreement is neither
    reachable nor wanted.
    """
    loss, n, hit = 0.0, 0, 0
    for inst in insts:
        lab = labels.get(inst.taskset_id)
        if lab is None:
            continue
        tf, ai, ao, cf = instance_tensors(inst, device)
        h, g = policy.encode(tf, ai, ao, cf)
        ep = Episode(inst)
        target = [inst.core_index[lab[t]] for t in inst.tasks]
        for k in range(inst.N):
            pair = torch.as_tensor(ep.features(), device=device)
            z = policy.logits(h[k], g, pair)
            loss = loss + F.cross_entropy(
                z.unsqueeze(0), torch.tensor([target[k]], device=device))
            hit += int(int(torch.argmax(z)) == target[k])
            ep.place(target[k])
        n += inst.N
    return loss / max(n, 1), hit / max(n, 1)


def reference(insts, name):
    """What a baseline scores on the validation set, computed once for context
    in the log."""
    fn = get(name)
    pen, perfect = [], []
    for inst in insts:
        p = inst.model.evaluate(fn(inst.model, inst.platform))["penalty"]
        pen.append(p["misses"] + p["budgets"])
        perfect.append(float(p["misses"] == 0 and p["budgets"] == 0))
    return sum(pen) / len(pen), 100 * sum(perfect) / len(perfect)


def validate(policy, insts, device, learn_order=False):
    policy.eval()
    hard, total, perfect = [], [], []
    with torch.no_grad():
        for inst in insts:
            s = rollout(policy, inst, greedy=True, device=device,
                        learn_order=learn_order)["score"]
            hard.append(s["hard"])
            total.append(s["total"])
            perfect.append(float(s["perfect"]))
    policy.train()
    return (sum(hard) / len(hard), sum(total) / len(total),
            100 * sum(perfect) / len(perfect))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=20000, help="optimiser steps")
    ap.add_argument("--batch", type=int, default=32, help="instances per step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--entropy", type=float, default=0.01,
                    help="entropy bonus, keeps the policy from collapsing early")
    ap.add_argument("--order", choices=["topo", "topo_strict", "deadline", "util"],
                    default="topo",
                    help="placement order. topo matches the greedy packers, and "
                         "falls back to criticality order on a single device "
                         "where the topological order carries no information. "
                         "topo_strict never does, for the ablation")
    ap.add_argument("--learn-order", action="store_true",
                    help="let the policy choose which ready task to place next, "
                         "not only where to put it. The same freedom bfd_crit "
                         "has. The warm start is always teacher forced in the "
                         "instance order, so this head is trained by RL alone")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3, help="graph layers")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--device", default="cpu",
                    help="cpu is usually faster here: the tensors are tiny and "
                         "kernel launch overhead dominates on a gpu")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads. 1 is fastest for tensors this small")
    ap.add_argument("--warm-from", default="bfd",
                    help="allocator whose saved allocations to imitate first, "
                         "or none")
    ap.add_argument("--warm-steps", type=int, default=2000)
    ap.add_argument("--warm-lr", type=float, default=1e-3,
                    help="supervised imitation wants a much larger step than "
                         "REINFORCE does")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--cache", type=int, default=3000, help="instances kept in memory")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N training task sets, spread evenly across "
                         "the pool. For smoke runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(MODELS))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    split = SPLITS / "train.txt"
    if not split.exists():
        raise SystemExit(f"no {split}. run scripts/generator/split_dataset.py first")
    ids = split.read_text().split()
    if not ids:
        raise SystemExit("empty train split. run scripts/generator/split_dataset.py first")

    # a stale split would otherwise show up a hundred lines later as an
    # unrelated complaint about missing labels
    on_disk = tasksets.count()
    if len(ids) < on_disk / 4:
        raise SystemExit(
            f"{split.name} lists {len(ids)} task sets but {on_disk} are on disk. "
            "The split predates the current dataset: rerun scripts/generator/split_dataset.py")

    if args.limit and args.limit < len(ids):
        # spread, not a prefix: the split is sorted by task set id, so the
        # first N would be platform_1 and platform_2 only
        ids = ids[::len(ids) // args.limit][:args.limit]

    # Validation is carved out of train, deterministically, since using
    # test_workload for model selection would leak it. Capped at a fifth of the
    # pool so a small --limit cannot swallow the training set.
    n_val = min(args.val_n, max(1, len(ids) // 5))
    val_ids = ids[::max(len(ids) // n_val, 1)][:n_val]
    val_set = set(val_ids)
    train_ids = [i for i in ids if i not in val_set]

    cache = Cache(args.cache, args.order)
    device = args.device
    policy = Policy(dim=args.dim, layers=args.layers, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"order": args.order, "learn_order": args.learn_order, "seed": args.seed,
            "task_features": list(envmod.TASK_FEATURES),
            "core_features": list(envmod.CORE_FEATURES),
            "pair_features": list(envmod.PAIR_FEATURES)}

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"train {len(train_ids)}   val {len(val_ids)}   "
          f"{n_params} parameters   device {device}   order {args.order}")

    print("\nbuilding the validation set and its reference scores...")
    val = [cache.get(i) for i in val_ids]
    ref = {}
    for name in ("bfd", "wfd"):
        ref[name] = reference(val, name)
        print(f"  {name:<5} pen_hard {ref[name][0]:8.1f}   perfect {ref[name][1]:5.1f}%")

    t0 = time.time()
    best = float("inf")

    if args.warm_from and args.warm_from != "none" and args.warm_steps > 0:
        labels = read_labels(args.warm_from)
        have = [i for i in train_ids if i in labels]
        if not have:
            raise SystemExit(f"{args.warm_from} has no allocation for any training "
                             "task set. Was it run with --per-cell?")
        print(f"\nwarm start: imitating {args.warm_from} on {len(have)} task sets, "
              f"{args.warm_steps} steps at lr {args.warm_lr}")
        # its own optimiser, since supervised learning wants a far larger step
        # than REINFORCE and sharing Adam's state would carry the supervised
        # moments into the policy gradient phase
        wopt = torch.optim.Adam(policy.parameters(), lr=args.warm_lr)
        # decayed to zero, because at a flat 1e-3 the optimiser keeps taking
        # full sized steps long after the loss stops moving
        wsched = torch.optim.lr_scheduler.CosineAnnealingLR(wopt, args.warm_steps)
        for step in range(1, args.warm_steps + 1):
            batch = [cache.get(i) for i in rng.sample(have, min(args.batch, len(have)))]
            loss, agree = warm_step(policy, batch, labels, device)
            wopt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            wopt.step()
            wsched.step()
            if step % args.log_every == 0 or step == args.warm_steps:
                el = time.time() - t0
                print(f"  warm {step}/{args.warm_steps}   loss {loss.item():.4f}   "
                      f"agree {100 * agree:5.1f}%   {hms(el)} elapsed, "
                      f"{hms(el / step * (args.warm_steps - step))} left")
            if step % args.val_every == 0:
                h, tt, pf = validate(policy, val, device, args.learn_order)
                print(f"    val  pen_hard {h:8.1f}   pen_total {tt:8.1f}   "
                      f"perfect {pf:5.1f}%")
        h, tt, pf = validate(policy, val, device, args.learn_order)
        print(f"  after warm start: pen_hard {h:.1f}  pen_total {tt:.1f}  "
              f"perfect {pf:.1f}%   (teacher {args.warm_from})")
        # imitation alone is the cloned, never improved row of the evaluation
        # table, which also makes --steps 0 a way to run behaviour cloning
        best = h
        save(policy, out / "rl_bc.pt", {**meta, "phase": "warm", "val_hard": h})
        save(policy, out / "rl.pt", {**meta, "phase": "warm", "val_hard": h})

    print(f"\nREINFORCE: {args.steps} steps x {args.batch} instances "
          f"= {args.steps * args.batch} episodes\n")

    t0 = time.time()
    run_pen, run_adv, run_ent = [], [], []

    n_batch = min(args.batch, len(train_ids))
    for step in range(1, args.steps + 1):
        batch = [cache.get(i) for i in rng.sample(train_ids, n_batch)]
        loss = 0.0

        for inst in batch:
            with torch.no_grad():
                base = rollout(policy, inst, greedy=True, device=device,
                               learn_order=args.learn_order)["score"]["total"]
            samp = rollout(policy, inst, greedy=False, device=device,
                           learn_order=args.learn_order)
            cost = samp["score"]["total"]

            # bounded to [-1, 1]: greedy at 5 against a sample at 500 would
            # otherwise give an advantage of -99 and drown out the batch
            scale = max(base, cost, 1.0)
            adv = (base - cost) / scale

            loss = loss - adv * samp["logp"] - args.entropy * samp["entropy"]
            run_pen.append(cost)
            run_adv.append(adv)
            run_ent.append(samp["entropy"].item())

        loss = loss / len(batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            el = time.time() - t0
            print(f"step {step}/{args.steps}   ep {step * args.batch}   "
                  f"pen {sum(run_pen) / len(run_pen):8.1f}   "
                  f"adv {sum(run_adv) / len(run_adv):+.3f}   "
                  f"ent {sum(run_ent) / len(run_ent):.3f}   "
                  f"{hms(el)} elapsed, {hms(el / step * (args.steps - step))} left")
            run_pen, run_adv, run_ent = [], [], []

        if step % args.val_every == 0 or step == args.steps:
            h, tt, pf = validate(policy, val, device, args.learn_order)
            tag = ""
            if h < best:
                best = h
                save(policy, out / "rl.pt", {**meta, "step": step, "val_hard": h})
                tag = "  <- best, saved"
            save(policy, out / "rl_last.pt", {**meta, "step": step, "val_hard": h})
            print(f"  val @ {step}   pen_hard {h:8.1f}   pen_total {tt:8.1f}   "
                  f"perfect {pf:5.1f}%   "
                  f"(bfd {ref['bfd'][0]:.1f}/{ref['bfd'][1]:.0f}%, "
                  f"wfd {ref['wfd'][0]:.1f}/{ref['wfd'][1]:.0f}%){tag}")

    print(f"\ndone in {hms(time.time() - t0)}")
    print(f"best validation pen_hard {best:.2f}")
    print(f"wrote {out / 'rl.pt'}")
    print("\nthe registry picks it up automatically. next:")
    print("  python scripts/run/run_allocators.py --only rl sa_rl --per-cell 10")
    print("  python scripts/analysis/analyse_allocations.py --common")


if __name__ == "__main__":
    main()
