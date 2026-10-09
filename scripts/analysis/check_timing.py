"""Run the timing model and show the tables, then the cross-link sanity check.

    python scripts/analysis/check_timing.py taskset_3_1_2_1

Step 2 is the tables: every task's response time, every core's load, the chain
against its budget, and the penalty they add up to. Step 3 moves one task across
a link and checks that the chain gets longer and two cores get busier. If that
does not happen the evaluator is wrong, and it exits non zero so it can be run
as a check rather than read as a report.

Takes a task set id. The platform comes from the task set, so there is no way to
ask for a pair that does not go together.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

import tasksets
from timing import Model

ROOT = Path(__file__).resolve().parent.parent.parent
PLATFORMS = ROOT / "data" / "platforms"


def round_robin(model, device):
    """Every task onto the cores of one device, in order. Nothing crosses."""
    cores = model.cores_of_device[device]
    return {tid: cores[i % len(cores)] for i, tid in enumerate(model.task)}


def task_table(model, res):
    print(f"\n{'task':<8}{'code':<20}{'core':<20}{'class':<10}"
          f"{'P':>6}{'D':>7}{'C_eff':>9}{'R':>9}  miss")
    for tid, r in sorted(res["tasks"].items(), key=lambda kv: kv[1]["core"]):
        t = model.task[tid]
        mark = "MISS" if r["miss"] else ""
        print(f"{tid:<8}{t['code']:<20}{r['core']:<20}{r['core_class']:<10}"
              f"{t['period_us']:>6}{r['deadline']:>7}{r['c_eff']:>9.1f}"
              f"{r['r']:>9.1f}{'' if r['r_bounded'] else '+'}  {mark}")


def core_table(model, res):
    print(f"\n{'core':<20}{'class':<10}{'cpu':<12}{'load':>8}   receives")
    for cid, load in res["cores"].items():
        c = model.core[cid]
        rx = res["receives"].get(cid, [])
        note = ", ".join(f"{r['c']}us/{r['period']}us" for r in rx)
        print(f"{cid:<20}{c['core_class']:<10}{c['cpu_type']:<12}{load:>8.3f}   {note}")


def chain_table(res):
    print(f"\n{'chain':<22}{'latency':>10}{'budget':>10}   path")
    for c in res["chains"]:
        b = "null" if c["budget_us"] is None else f"{c['budget_us']:.0f}"
        print(f"{c['src'] + ' -> ' + c['dst']:<22}{c['latency_us']:>10.1f}"
              f"{'' if c['exact'] else '+'}{b:>10}"
              f"   {' > '.join(c['path'])}{'   VIOLATED' if c['violated'] else ''}")
    print("   + means a task on the path missed, so its R stopped early and the "
          "number is a lower bound")


def penalty_line(res):
    p = res["penalty"]
    print(f"\npenalty  total {p['total']:.2f}  = misses {p['misses']:.2f}"
          f" + budgets {p['budgets']:.2f} + preference {p['preference']:.2f}")


def load(taskset_id):
    """One task set and the platform it was generated for.

    The platform is named inside the task set rather than passed alongside it,
    so there is no way to ask for a mismatched pair.
    """
    got = list(tasksets.stream([taskset_id]))
    if not got:
        raise SystemExit(f"no task set {taskset_id}. try one of:\n  "
                         + ", ".join(tasksets.select(per_cell=1)[:6]))
    taskset = got[0][1]
    pid = taskset["generator"]["platform_id"]
    platform = json.loads((PLATFORMS / f"{pid}.json").read_text())
    return taskset, platform


def main(taskset_id):
    taskset, platform = load(taskset_id)
    model = Model(taskset, platform)

    main_dev = platform["devices"][0]["device_id"]
    alloc = round_robin(model, main_dev)

    print("=" * 100)
    print(f"STEP 2   {taskset['taskset_id']} on {platform['platform_id']}, "
          f"everything on {main_dev}, nothing crosses")
    print("=" * 100)
    base = model.evaluate(alloc)
    task_table(model, base)
    core_table(model, base)
    chain_table(base)
    penalty_line(base)

    if len(platform["devices"]) < 2:
        print("\nsingle device platform, no link to cross. step 3 skipped.")
        return

    # step 3: move a task that is in the middle of the chain, so it has both an
    # incoming and an outgoing edge, onto the second device
    path = base["chains"][0]["path"]
    moved = path[len(path) // 2]
    other_dev = platform["devices"][1]["device_id"]
    target = model.cores_of_device[other_dev][-1]

    alloc2 = dict(alloc)
    alloc2[moved] = target

    print("\n" + "=" * 100)
    print(f"STEP 3   move {moved} ({model.task[moved]['code']}) "
          f"from {alloc[moved]} to {target}")
    print("=" * 100)
    after = model.evaluate(alloc2)
    task_table(model, after)
    core_table(model, after)
    chain_table(after)
    penalty_line(after)

    print("\n--- what changed ---")
    d_chain = after["chains"][0]["latency_us"] - base["chains"][0]["latency_us"]
    print(f"chain latency  {base['chains'][0]['latency_us']:.1f} -> "
          f"{after['chains'][0]['latency_us']:.1f}   ({d_chain:+.1f})")

    busier = [(cid, base["cores"][cid], after["cores"][cid])
              for cid in model.core
              if after["cores"][cid] > base["cores"][cid] + 1e-9]
    for cid, b, a in busier:
        print(f"busier         {cid:<20} {b:.3f} -> {a:.3f}   ({a - b:+.3f})")

    ok_chain = d_chain > 0
    ok_cores = len(busier) >= 2
    print(f"\nchain got longer: {ok_chain}    at least two cores got busier: {ok_cores}")
    if not (ok_chain and ok_cores):
        print("SANITY CHECK FAILED")
        sys.exit(1)
    print("sanity check passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("taskset", nargs="?",
                    help="task set id. Defaults to the first one in the store")
    args = ap.parse_args()
    main(args.taskset or tasksets.select(per_cell=1)[0])
