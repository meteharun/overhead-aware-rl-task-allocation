"""Split the dataset into train and two test pools.

    python scripts/generator/split_dataset.py
    python scripts/generator/split_dataset.py --test-frac 0.2 --holdout 10

Writes data/tasksets/{train,test_workload,test_hardware}.txt, one task set id per
line. Nothing is moved or copied: a split is a list of names.

Three pools, because they answer different questions:

  train           what the policy learns from
  test_workload   unseen task sets on platforms it has seen. Does it generalise
                  to new workloads?
  test_hardware   every task set on a platform held out of training entirely.
                  Does it generalise to hardware it has never seen?

Stratified over every (platform, coupling, load) cell, so train and test have
identical composition and no cell is missing from either.
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import tasksets
from gen_platform import EVAL_ONLY

OUT = tasksets.DIR


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="share of each cell held out as unseen workloads")
    ap.add_argument("--holdout", type=int, nargs="+", default=list(EVAL_ONLY),
                    help="platform numbers kept out of training entirely. "
                         "Defaults to EVAL_ONLY in gen_platform.py, so the two "
                         "cannot drift apart")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ids = tasksets.ids()
    if not ids:
        raise SystemExit("no task sets found. run scripts/generator/gen_dataset.py first")

    # the id carries platform, coupling and load, so the cells can be built
    # without opening a single record
    cells = defaultdict(list)
    for tid in ids:
        p, c, l, _ = tasksets.parse_id(tid)
        cells[(f"platform_{p}", c, l)].append(tid)

    rng = random.Random(args.seed)
    held = {f"platform_{n}" for n in args.holdout}
    train, test_workload, test_hardware = [], [], []

    for (pid, _, _), ids in sorted(cells.items()):
        ids = sorted(ids)
        if pid in held:
            test_hardware += ids
            continue
        rng.shuffle(ids)
        k = round(len(ids) * args.test_frac)
        test_workload += ids[:k]
        train += ids[k:]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, ids in (("train", train), ("test_workload", test_workload),
                      ("test_hardware", test_hardware)):
        (OUT / f"{name}.txt").write_text("\n".join(sorted(ids)) + "\n")
        print(f"{name:<16}{len(ids):>7} task sets")

    overlap = set(train) & (set(test_workload) | set(test_hardware))
    print(f"\noverlap between pools: {len(overlap)}  (must be 0)")
    print(f"held out of training: {', '.join(sorted(held, key=lambda p: int(p.split('_')[1])))}")
    print(f"wrote {OUT}")
    # the set key is written when an allocation is, so a new split means the
    # allocations have to be produced again rather than relabelled
    print("\nnext: python scripts/run/run_allocators.py --eval")


if __name__ == "__main__":
    main()
