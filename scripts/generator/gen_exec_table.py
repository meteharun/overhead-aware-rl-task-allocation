"""Generate data/catalogues/exec_table.json.

    python scripts/generator/gen_exec_table.py

A55 comes from the board. `board/measure_wcet.c` runs each STATCOM task alone
on every core of the i.MX95 under SCHED_FIFO with memory locked and reports the
distribution per core class. `scripts/generator/map_measured_to_catalogue.py`
turns that into data/catalogues/a55_exec_measured.json, which is the A55 column.

A78 and Kryo_Gen6 have no board, so their columns are drawn: each (code, cpu)
pair uniformly in a fraction of the code's period, seed below. They are marked
as such in the output and in Table 4.1 of the thesis. No cpu type is assumed
faster than another.

Which statistic becomes the execution time is a choice and it is recorded in the
output. The default is p50: the model multiplies the whole response window by the
core-class tail factor, so taking a tail percentile here would charge the tail
twice.
"""

import argparse
import json
import random
from pathlib import Path

SEED = 20260820
CPU_TYPES = ["A55", "A78", "Kryo_Gen6"]
MEASURED = "A55"
# fraction of the period, for the cpu types with no board. Up to 0.50 lets a
# single task eat half a core, which leaves the allocator no room to choose.
LO, HI = 0.10, 0.30

ROOT = Path(__file__).resolve().parent.parent.parent
CAT = ROOT / "data" / "catalogues"
MEASUREMENT = CAT / "a55_exec_measured.json"


def load_measured(path, statistic):
    """The A55 column, from the board run.

    Fails loudly rather than falling back to a draw, since nothing downstream
    could tell a substituted random number from a measurement.
    """
    if not path.exists():
        raise SystemExit(
            f"no {path}\n"
            "Run the board campaign first:\n"
            "  scp -r board/ <board>:~/ && ssh <board>\n"
            "  cd board && sudo ./run_measurements.sh\n"
            f"then copy a55_exec_raw.json here as {path.name}")
    raw = json.loads(path.read_text())
    got = raw.get("measurements") or {}
    if not got:
        raise SystemExit(f"{path} has no measurements block")
    out = {}
    for code, row in got.items():
        if statistic not in row:
            raise SystemExit(f"{path}: {code} has no '{statistic}'. "
                             f"Available: {', '.join(sorted(row))}")
        out[code] = round(float(row[statistic]), 2)
    return out, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--statistic", default="p50",
                    choices=("p50", "p90", "p99", "p999", "max", "mean"),
                    help="which point of the measured distribution becomes the "
                         "execution time. Default p50: the model multiplies the "
                         "whole response window by the core-class tail factor, "
                         "so taking a tail percentile here would charge the "
                         "tail twice")
    ap.add_argument("--measurement", default=str(MEASUREMENT))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing execution table. Without this "
                         "the generator refuses, because every allocation in "
                         "data/allocations was computed against the table on "
                         "disk and replacing it invalidates all of them")
    args = ap.parse_args()

    cat = json.loads((CAT / "task_catalogue.json").read_text())
    measured, raw = load_measured(Path(args.measurement), args.statistic)

    missing = [e["code"] for e in cat["codes"] if e["code"] not in measured]
    if missing:
        raise SystemExit(f"the board run is missing {len(missing)} codes: "
                         + ", ".join(missing[:8])
                         + ("..." if len(missing) > 8 else ""))

    rng = random.Random(SEED)
    exec_us = {}
    for entry in cat["codes"]:
        row = {}
        for cpu in CPU_TYPES:
            # The draw happens for every cpu type, including the measured one,
            # and the measured column then replaces it. Skipping the call for
            # A55 instead would shift the random stream by one position per
            # code, so A78 and Kryo_Gen6 would come out different from the
            # table every allocation in data/allocations was computed against.
            drawn = round(rng.uniform(LO, HI) * entry["period_us"], 2)
            row[cpu] = measured[entry["code"]] if cpu == MEASURED else drawn
        exec_us[entry["code"]] = row

    out = {
        "cpu_types": CPU_TYPES,
        "measured": {
            "cpu_type": MEASURED,
            "statistic": args.statistic,
            "source": Path(args.measurement).name,
            "environment": raw.get("environment", {}),
            "iterations": raw.get("iterations"),
            "clock_overhead_us": raw.get("clock_overhead_us"),
        },
        "synthetic": {
            "cpu_types": [c for c in CPU_TYPES if c != MEASURED],
            "seed": SEED,
            "rule": f"drawn uniformly in [{LO}, {HI}] of the code period",
        },
        "keyed_by": "code name, so two tasks running the same code share one row",
        "exec_us": exec_us,
    }
    path = CAT / "exec_table.json"
    if path.exists() and not args.force:
        raise SystemExit(
            f"{path} already exists.\n"
            "Every allocation in data/allocations was computed against that "
            "table, so replacing it invalidates all of them and every number "
            "in the results chapters. Pass --force if that is what you want, "
            "and rerun scripts/run/run_allocators.py afterwards.")
    path.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'id':<5}{'code':<20}{'P':>6}" + "".join(f"{c:>12}" for c in CPU_TYPES)
          + "   fastest")
    fastest = dict.fromkeys(CPU_TYPES, 0)
    for entry in cat["codes"]:
        r = exec_us[entry["code"]]
        win = min(CPU_TYPES, key=lambda c: r[c])
        fastest[win] += 1
        print(f"{entry['id']:<5}{entry['code']:<20}{entry['period_us']:>6}"
              + "".join(f"{r[c]:>12.2f}" for c in CPU_TYPES)
              + f"   {win}")
    print(f"\nwrote {path}")
    print(f"{MEASURED}: measured, {args.statistic} of "
          f"{raw.get('iterations', '?')} iterations on "
          f"{raw.get('environment', {}).get('kernel', 'the board')}")
    print("others: drawn, seed " + str(SEED))
    print("fastest per code: " + ", ".join(f"{c} {n}" for c, n in fastest.items()))


if __name__ == "__main__":
    main()
