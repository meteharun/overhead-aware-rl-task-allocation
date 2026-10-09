"""Turn the board campaign into data/catalogues/a55_exec_measured.json.

    python scripts/generator/map_measured_to_catalogue.py

The campaign measures 25 STATCOM tasks. The benchmark catalogue has 50 codes,
derived from the same workload with variants that change the cost while keeping
the functional role, so each code has to be matched to a measured task.

The matching rule, which is the whole of this file:

  1. A code can only take a task of its own period. Period is what sets the
     criticality band and the deadline, so crossing periods would break the
     thing the catalogue is built around.
  2. Within a period, codes are ordered light, medium, heavy by their declared
     cost_class, and the measured tasks are ordered by cost. The i-th of n codes
     takes the (i * m / n)-th of m tasks.

So a light 100 us code gets the cheapest 100 us task measured and a heavy one
gets the dearest, and the spread the catalogue declares survives into the
numbers. Each entry records which task it came from, so any row can be traced
back to the campaign.

Source. By default this reads board/wcets_isolated.jsonl, the campaign
already in the repository: 25 tasks, each run back-to-back with a warm cache on
all six cores, 115k to 12M samples apiece. `--raw` reads the output of
board/measure_wcet.c instead, for a fresh run.

Statistics are in microseconds, which is what the execution table and the timing
model use. The campaign records nanoseconds.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CAT = ROOT / "data" / "catalogues"
CAMPAIGN = ROOT / "board" / "wcets_isolated.jsonl"
OUT = CAT / "a55_exec_measured.json"

COST_RANK = {"light": 0, "medium": 1, "heavy": 2}


def read_campaign(path):
    """The 25 measured tasks, as microseconds."""
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "task": r["task"],
                "period_us": r["period_us"],
                "tier": r["tier"],
                "n_samples": r["n_samples"],
                "min": r["min_ns"] / 1000.0,
                "p50": r["nominal_ns"] / 1000.0,
                "p90": r["p90_ns"] / 1000.0,
                "p99": r["p99_ns"] / 1000.0,
                "p999": r["p999_ns"] / 1000.0,
                "max": r["max_ns"] / 1000.0,
                "mean": r["mean_ns"] / 1000.0,
                "p999_general": r["p999_general_ns"] / 1000.0,
                "p999_isolated": r["p999_isolated_ns"] / 1000.0,
            })
    return rows


def read_raw(path):
    """The output of board/measure_wcet.c, as microseconds."""
    raw = json.loads(path.read_text())
    rows = []
    for name, m in raw["measurements"].items():
        cls = m.get("per_class", {})
        gen = cls.get("general", {})
        iso = cls.get("isolated", {})
        cores = list(m["per_core"].values())
        rows.append({
            "task": name,
            "period_us": m["period_us"],
            "tier": m["tier"],
            "n_samples": raw.get("iterations_per_core", 0) * len(cores),
            "min": m["min"] / 1000.0,
            "p50": m["nominal_ns"] / 1000.0,
            "p90": max(c["p90"] for c in cores) / 1000.0,
            "p99": max(c["p99"] for c in cores) / 1000.0,
            "p999": max(c["p999"] for c in cores) / 1000.0,
            "max": m["max"] / 1000.0,
            "mean": m["mean"] / 1000.0,
            "p999_general": gen.get("p999", 0.0) / 1000.0,
            "p999_isolated": iso.get("p999", 0.0) / 1000.0,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default=str(CAMPAIGN),
                    help="wcets_isolated.jsonl from the original campaign")
    ap.add_argument("--raw", default=None,
                    help="a55_exec_raw.json from a fresh board/measure_wcet.c run")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    if args.raw:
        src = Path(args.raw)
        tasks = read_raw(src)
    else:
        src = Path(args.campaign)
        if not src.exists():
            raise SystemExit(f"no {src}")
        tasks = read_campaign(src)

    cat = json.loads((CAT / "task_catalogue.json").read_text())
    codes = cat["codes"]

    by_period = {}
    for t in tasks:
        by_period.setdefault(t["period_us"], []).append(t)
    for p in by_period:
        by_period[p].sort(key=lambda t: t["p50"])

    missing = sorted({c["period_us"] for c in codes} - set(by_period))
    if missing:
        raise SystemExit(f"no measured task at period(s) {missing}")

    out, used = {}, {}
    for period in sorted(by_period):
        pool = by_period[period]
        group = sorted((c for c in codes if c["period_us"] == period),
                       key=lambda c: (COST_RANK.get(c["cost_class"], 1), c["id"]))
        n, m = len(group), len(pool)
        for i, c in enumerate(group):
            t = pool[i * m // n]
            used.setdefault(t["task"], []).append(c["code"])
            out[c["code"]] = {
                "min": round(t["min"], 3),
                "p50": round(t["p50"], 3),
                "p90": round(t["p90"], 3),
                "p99": round(t["p99"], 3),
                "p999": round(t["p999"], 3),
                "max": round(t["max"], 3),
                "mean": round(t["mean"], 3),
                "p999_general": round(t["p999_general"], 3),
                "p999_isolated": round(t["p999_isolated"], 3),
                "source_task": t["task"],
                "cost_class": c["cost_class"],
                "n_samples": t["n_samples"],
            }

    doc = {
        "cpu_type": "A55",
        "device": "Verdin i.MX95, 6 x Cortex-A55",
        "unit": "microseconds",
        "source": src.name,
        "method": ("back-to-back, warm cache, alone on the core, on all six "
                   "cores with the per-core results merged. p50 is the median "
                   "across cores; p999_general covers the cores that take "
                   "interrupts and p999_isolated the isolcpus cores"),
        "mapping": ("25 measured tasks onto 50 catalogue codes, matched within "
                    "period and ordered by the catalogue's cost_class. "
                    "source_task on each entry names the task it came from"),
        "measured_tasks": len(tasks),
        "codes": len(out),
        "measurements": out,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"{len(tasks)} measured tasks -> {len(out)} catalogue codes")
    print(f"{'period':>7}{'codes':>7}{'tasks':>7}   p50 range (us)")
    for period in sorted(by_period):
        g = [v for k, v in out.items()
             if next(c for c in codes if c["code"] == k)["period_us"] == period]
        lo = min(v["p50"] for v in g)
        hi = max(v["p50"] for v in g)
        print(f"{period:>7}{len(g):>7}{len(by_period[period]):>7}   "
              f"{lo:.2f} - {hi:.2f}")
    spare = [t["task"] for t in tasks if t["task"] not in used]
    if spare:
        print(f"\nmeasured but unused: {', '.join(spare)}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
