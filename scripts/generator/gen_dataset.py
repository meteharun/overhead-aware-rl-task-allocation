"""Generate the whole task set grid.

    python scripts/generator/gen_dataset.py                  20 x 4 x 5 x 40 = 16000
    python scripts/generator/gen_dataset.py --per-cell 5     a quick probe
    python scripts/generator/gen_dataset.py --platforms 3 4  only those two

Writes one file, data/tasksets/tasksets.jsonl, plus its index. See tasksets.py
for why it is one file and not one per instance.

Every cell is one (platform, coupling, load) combination. Coupling and load are
jittered inside the cell, so its instances are a region rather than 40 copies.
"""

import argparse
import time

import gen_taskset as G
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
import tasksets


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-cell", type=int, default=40)
    ap.add_argument("--platforms", type=int, nargs="*", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="start a new store instead of adding to the one there")
    args = ap.parse_args()

    platforms = args.platforms or sorted(
        int(p.stem.split("_")[1])
        for p in (tasksets.ROOT / "data" / "platforms").glob("platform_*.json"))
    couplings, loads = sorted(G.COUPLING), sorted(G.LOAD)
    total = len(platforms) * len(couplings) * len(loads) * args.per_cell

    print(f"{len(platforms)} platforms x {len(couplings)} coupling x {len(loads)} load "
          f"x {args.per_cell} = {total} task sets\n")

    cache, done, t0 = {}, 0, time.time()
    # one writer for the whole grid, so the store is opened once
    with tasksets.Writer(fresh=args.fresh) as store:
        for p in platforms:
            for c in couplings:
                for l in loads:
                    for n in range(1, args.per_cell + 1):
                        G.generate(p, c, l, n, quiet=True, cache=cache, store=store)
                        done += 1
                        if done % 250 == 0 or done == total:
                            el = time.time() - t0
                            print(f"  {done}/{total}   {el:.0f}s elapsed, "
                                  f"{el / done * (total - done):.0f}s left")

    print(f"\ndone in {time.time() - t0:.0f}s -> {tasksets.STORE}")
    print(f"{tasksets.count()} task sets in the store")
    print("next: python scripts/generator/split_dataset.py")


if __name__ == "__main__":
    main()
