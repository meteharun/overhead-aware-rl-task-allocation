#!/usr/bin/env python3
"""Every figure in the thesis, by its number in the write-up.

    python scripts/fig/generate_figs.py                 list them and stop
    python scripts/fig/generate_figs.py 1 3 5           just those three
    python scripts/fig/generate_figs.py 13-20           a range
    python scripts/fig/generate_figs.py all             the lot
    python scripts/fig/generate_figs.py 15 --collect    copy it into out/writing too

Numbers follow the order the figures appear in the thesis, so the number here
and the number in the document are the same thing. `--list` prints the table
with the command behind each one.

Each figure keeps its own file under scripts/fig, since each loads different
data and has its own options. This runs them in order with the arguments the
thesis version needs and reports which output files changed. It holds no copy
of the drawing code, so the driver and a figure cannot disagree.

Run it with the interpreter that has matplotlib, since it hands its own
executable to each child process:

    /c/Users/<you>/.conda/envs/rl/python.exe scripts/fig/generate_figs.py all

Four of the twenty are not produced by any script. Figure 1 is exported by hand
from architecture.drawio, and figures 7 to 9 are TikZ drawn in the document
itself. They are listed so the numbering matches the thesis, and asking for one
says where it comes from instead of failing.
"""

import argparse
import shutil
import subprocess
import sys
import time
from collections import namedtuple
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out"
WRITING = OUT / "writing"

# name     the file the thesis includes
# cmd      argv after the interpreter, relative to the repo root. None means
#          nothing here builds it
# produces where the command actually writes, relative to out/
Fig = namedtuple("Fig", "n name cmd produces note")

FIGURES = [
    Fig(1, "architecture.drawio.png", None, None,
        "exported by hand from architecture.drawio in the repo root"),
    Fig(2, "example_taskset.png",
        ["scripts/fig/gen_example_figs.py"], "writing/example_taskset.png", ""),
    Fig(3, "example_platform.png",
        ["scripts/fig/gen_example_figs.py"], "writing/example_platform.png",
        "same run as figure 2"),
    Fig(4, "penalty-example.png",
        ["scripts/fig/viz_penalty_example.py"], "writing/penalty-example.png", ""),
    Fig(5, "model_io.png",
        ["scripts/fig/viz_model_io.py"], "writing/model_io.png", ""),
    Fig(6, "greedy_pair.png",
        ["scripts/fig/viz_greedy_pair.py"], "writing/greedy_pair.png", ""),
    Fig(7, "learned-representation.tex", None, None,
        "TikZ, drawn in the document"),
    Fig(8, "learned-policy-architecture.tex", None, None,
        "TikZ, drawn in the document"),
    Fig(9, "annealing-workflow.tex", None, None,
        "TikZ, drawn in the document"),
    Fig(10, "anneal_example.png",
        ["scripts/fig/viz_anneal_example.py"], "writing/anneal_example.png", ""),
    Fig(11, "platform_13.png",
        ["scripts/visualization/viz_platform.py", "13"],
        "platforms/platform_13.png", "lands in out/platforms, not out/writing"),
    Fig(12, "taskset_10_3_3_3.png",
        ["scripts/visualization/viz_taskset.py", "10_3_3_3"],
        "tasksets/taskset_10_3_3_3.png",
        "lands in out/tasksets, not out/writing"),
    Fig(13, "onepass_quality.png",
        ["scripts/fig/viz_onepass.py"], "writing/onepass_quality.png", ""),
    Fig(14, "portfolio_rl_penalty_difference.png",
        ["scripts/fig/viz_portfolio.py",
         "--name", "portfolio_rl_penalty_difference"],
        "writing/portfolio_rl_penalty_difference.png",
        "the script's own default name is onepass-portfolio-vs-rl"),
    Fig(15, "budget_curve.png",
        ["scripts/fig/viz_budget.py"], "writing/budget_curve.png", ""),
    Fig(16, "offline-generalisation.png",
        ["scripts/fig/viz_generalisation.py"],
        "writing/offline-generalisation.png", ""),
    Fig(17, "offline-difficulty.png",
        ["scripts/fig/viz_difficulty.py"], "writing/offline-difficulty.png", ""),
    Fig(18, "offline-behaviour.png",
        ["scripts/fig/viz_behaviour.py"], "writing/offline-behaviour.png", ""),
    Fig(19, "repair-example.png",
        ["scripts/fig/viz_repair_concept.py"], "writing/repair-example.png", ""),
    Fig(20, "repair-rebuild-budget-mean_2.png",
        ["scripts/fig/viz_repair.py", "--only", "budget"],
        "writing/repair-rebuild-budget-mean.png",
        "thesis copy carries a _2 suffix; --collect renames it"),
]

BY_N = {f.n: f for f in FIGURES}


def parse_selection(args):
    """Numbers, ranges and `all`, in thesis order and without repeats."""
    if any(a.lower() == "all" for a in args):
        return [f.n for f in FIGURES]
    want = []
    for a in args:
        if "-" in a and not a.startswith("-"):
            lo, _, hi = a.partition("-")
            try:
                lo, hi = int(lo), int(hi)
            except ValueError:
                raise SystemExit(f"not a range: {a!r}")
            want += list(range(lo, hi + 1))
        else:
            try:
                want.append(int(a))
            except ValueError:
                raise SystemExit(f"not a figure number: {a!r}")
    bad = [n for n in want if n not in BY_N]
    if bad:
        raise SystemExit(f"no figure {', '.join(map(str, bad))}. "
                         f"They run 1 to {max(BY_N)}.")
    return [f.n for f in FIGURES if f.n in set(want)]


def show(selected=None):
    print(f"{'':>3}  {'figure in the thesis':<42}{'built by'}")
    for f in FIGURES:
        mark = " " if selected is None or f.n in selected else "."
        cmd = " ".join(f.cmd) if f.cmd else f"(not a script) {f.note}"
        cmd = cmd.replace("scripts/fig/", "fig/").replace(
            "scripts/visualization/", "visualization/")
        print(f"{mark}{f.n:>2}  {f.name:<42}{cmd}")


def run(numbers, collect=False, dry=False):
    seen, plan = set(), []
    for n in numbers:
        f = BY_N[n]
        if f.cmd is None:
            plan.append((f, None))
            continue
        key = tuple(f.cmd)
        plan.append((f, None if key in seen else key))
        seen.add(key)

    width = max(len(f.name) for f, _ in plan)
    made, skipped, failed = [], [], []

    for f, key in plan:
        if f.cmd is None:
            print(f"  {f.n:>2} {f.name:<{width}}  not built here: {f.note}")
            skipped.append(f)
            continue
        target = OUT / f.produces
        if key is None:
            print(f"  {f.n:>2} {f.name:<{width}}  came with figure "
                  f"{next(g.n for g, k in plan if k == tuple(f.cmd))}")
            made.append((f, target))
            continue

        if dry:
            print(f"  {f.n:>2} {f.name:<{width}}  {' '.join(f.cmd)}")
            continue

        before = target.stat().st_mtime if target.exists() else 0
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, *f.cmd], cwd=ROOT,
                           capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()
            print(f"  {f.n:>2} {f.name:<{width}}  FAILED after {dt:.1f}s")
            for line in tail[-4:]:
                print(f"      {line}")
            failed.append(f)
            continue
        if not target.exists():
            print(f"  {f.n:>2} {f.name:<{width}}  ran, but {target} is missing")
            failed.append(f)
            continue
        fresh = target.stat().st_mtime > before
        print(f"  {f.n:>2} {f.name:<{width}}  "
              f"{'wrote' if fresh else 'unchanged'} {f.produces}  ({dt:.1f}s)")
        made.append((f, target))

    if collect and not dry:
        WRITING.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f, target in made:
            dest = WRITING / f.name
            if target.exists() and target.resolve() != dest.resolve():
                shutil.copy2(target, dest)
                copied += 1
        if copied:
            print(f"\ncollected {copied} file(s) into {WRITING} under their "
                  f"thesis names")

    print(f"\n{len(made)} built, {len(skipped)} not built here, "
          f"{len(failed)} failed")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*",
                    help="figure numbers, ranges like 13-20, or 'all'")
    ap.add_argument("--list", action="store_true",
                    help="print the table and stop")
    ap.add_argument("--collect", action="store_true",
                    help="copy every built figure into out/writing under the "
                         "name the thesis includes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    if args.list or not args.figures:
        show()
        if not args.figures:
            print(f"\nPick some: python {Path(__file__).name} 1 3 5 | 13-20 | all")
        return 0
    return run(parse_selection(args.figures), args.collect, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
