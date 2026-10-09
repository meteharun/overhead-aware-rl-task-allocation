#!/usr/bin/env python3
"""Turn a deepened run into ordinary allocation files.

    python scripts/run/deepen_to_allocations.py
    python scripts/run/deepen_to_allocations.py --only sa_rl
    python scripts/run/deepen_to_allocations.py --out-dir out/runs   # keep them aside

Writes data/allocations/<allocator>_deep.jsonl, one per allocator in the
deepened run, each with 6,400 rows in exactly the shape `run_allocators.py`
writes, so every analysis and figure script reads them without changes.

`out/runs/deepen_all.jsonl` is not in that shape: it has `taskset` rather than
`taskset_id`, no `summary` block, and its curve is separate from the one the
4,000-evaluation run recorded. This stitches the two together and re-scores
every allocation with the timing model, so the summary is computed the same way
as in every other file rather than copied across.

The deepened run only visited instances that still had a timing penalty after
4,000 evaluations. The rest keep their 4,000-evaluation allocation and have
their curve extended flat, which is what a longer search would have returned
for them anyway, since best-so-far is monotone and theirs was already zero.

These files are a longer run of the same search, not a different allocator. The
thesis compares starting points at a matched 4,000 evaluations, and
`viz_budget.py` leaves them out of that chart on purpose.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import splits
import tasksets
from progress import Progress
from run_allocators import summarise
from timing import Model, load_catalogues

ALLOC = ROOT / "data" / "allocations"
PLATFORMS = ROOT / "data" / "platforms"
DEEP = ROOT / "out" / "runs" / "deepen_all.jsonl"
SUFFIX = "_deep"


def read_deep(path):
    """Last row per (allocator, task set). A later row supersedes an earlier one,
    which is how --extend records a pair it carried further."""
    if not path.exists():
        raise SystemExit(f"no {path}. Run scripts/run/deepen_all.py first.")
    out = defaultdict(dict)
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue            # a line cut in half by a crash
            out[r["allocator"]][r["taskset"]] = r
    return out


def stitch(base_curve, deep_curve):
    """The 4,000-evaluation curve followed by the deepened one.

    Both are keyed by total budget, so they concatenate. The deepened points
    were recorded by the search itself at a fixed ladder, so there is no gap.
    """
    curve = dict(base_curve)
    curve.update(deep_curve or {})
    return curve


def flat(base_curve, budgets):
    """For an instance the deepening never visited: hold the 4,000 point."""
    curve = dict(base_curve)
    if not curve:
        return curve
    last = curve[max(curve, key=lambda k: int(k))]
    for b in budgets:
        curve[str(b)] = dict(last)
    return curve


def convert(name, deep_rows, out_dir, exec_us, hw):
    src = ALLOC / f"{name}.jsonl"
    if not src.exists():
        raise SystemExit(f"no {src}")
    rows = [json.loads(l) for l in src.open()]

    ladder = sorted({int(b) for r in deep_rows.values() for b in r["curve"]})
    out = out_dir / f"{name}{SUFFIX}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    platforms, bar = {}, Progress(len(rows), f"{name} instances")
    changed = n_perfect = 0

    with out.open("w") as fh:
        for i, r in enumerate(rows, start=1):
            tid, pid = r["taskset_id"], r["platform_id"]
            if pid not in platforms:
                platforms[pid] = json.loads((PLATFORMS / f"{pid}.json").read_text())
            taskset = tasksets.load(tid)
            model = Model(taskset, platforms[pid], exec_us, hw)

            deep = deep_rows.get(tid)
            alloc = deep["allocation"] if deep else r["allocation"]
            res = model.evaluate(alloc)
            base_curve = r["summary"].get("sa_curve") or {}

            extra = {k: v for k, v in r["summary"].items() if k.startswith("sa_")}
            if deep:
                changed += 1
                extra.update({
                    "sa_best_penalty": round(res["penalty"]["total"], 4),
                    "sa_best_hard": deep["hard"],
                    "sa_evals": 4000 + deep["evals_total"],
                    "sa_curve": stitch(base_curve, deep["curve"]),
                    "deep_evals": deep["evals_total"],
                    "deep_rounds": deep["rounds_total"],
                    "deep_seconds": deep["seconds"],
                    "deep_solved": deep["solved"],
                })
            else:
                extra["sa_curve"] = flat(base_curve, ladder)
                extra.update({"deep_evals": 0, "deep_rounds": 0,
                              "deep_seconds": 0.0, "deep_solved": None})

            summary = summarise(model, alloc, res, extra=extra,
                                ms=r["summary"].get("alloc_ms"))
            n_perfect += summary["perfect"]
            fh.write(json.dumps({
                "taskset_id": tid,
                "set": r.get("set", splits.set_of(tid)),
                "platform_id": pid,
                "allocator": f"{name}{SUFFIX}",
                "coupling_level": r["coupling_level"],
                "load_level": r["load_level"],
                "n_independent": r["n_independent"],
                "edges_per_task": r["edges_per_task"],
                "allocation": alloc,
                "summary": summary,
            }) + "\n")
            bar.tick(i, f"{n_perfect:,} perfect")
    bar.close()
    return out, len(rows), changed, n_perfect


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep", default=str(DEEP),
                    help="the deepened run to convert")
    ap.add_argument("--only", nargs="+", default=None, metavar="ALLOCATOR",
                    help="just these, e.g. sa_rl. Default: every one in the run")
    ap.add_argument("--out-dir", default=str(ALLOC),
                    help="where the converted files go. Anything inside "
                         "data/allocations is picked up by every script that "
                         "globs that directory")
    args = ap.parse_args()

    deep = read_deep(Path(args.deep))
    names = args.only or sorted(deep)
    missing = [n for n in names if n not in deep]
    if missing:
        raise SystemExit(f"not in the deepened run: {', '.join(missing)}. "
                         f"It has {', '.join(sorted(deep))}")

    exec_us, hw = load_catalogues()
    out_dir = Path(args.out_dir)
    results = []
    for name in names:
        results.append((name, *convert(name, deep[name], out_dir, exec_us, hw)))

    print(f"\n{'allocator':<16}{'rows':>8}{'deepened':>10}{'perfect':>10}{'rate':>8}")
    for name, path, n, changed, perfect in results:
        print(f"{name + SUFFIX:<16}{n:>8,}{changed:>10,}{perfect:>10,}"
              f"{100 * perfect / n:>7.1f}%")
    print(f"\nwrote {len(results)} file(s) to {out_dir}")
    if out_dir.resolve() == ALLOC.resolve():
        print("\nInside data/allocations, so the scripts that glob it will see "
              "them:\n"
              "  fig/viz_budget_deep.py   draws the extended curves\n"
              "  fig/viz_budget.py        skips them, by name\n"
              "  analysis/analyse_allocations.py  adds a row per file\n"
              "Use --out-dir to keep them out of the way instead.")


if __name__ == "__main__":
    main()
