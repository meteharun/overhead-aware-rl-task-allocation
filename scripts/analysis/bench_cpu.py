#!/usr/bin/env python3
"""Characterise this machine, so timings taken on it can be moved to another one.

    python scripts/analysis/bench_cpu.py                          full run, writes out/bench/
    python scripts/analysis/bench_cpu.py --reps 5000
    python scripts/analysis/bench_cpu.py --only kernel project
    python scripts/analysis/bench_cpu.py --compare out/bench/a55.json

Everything in this thesis was timed on a laptop. The target is an A55. Nothing
in the results is wrong because of that, but every millisecond quoted is a
laptop millisecond, and the thesis has to say what that means.

This is the instrument, not the answer: it profiles one machine, and a
conversion factor needs two. Run it unchanged on an A55 and pass the result to
`--compare` for the other half.

`hardware_catalogue.json` carries the only real A55 numbers in the project,
measured on the i.MX95: wake 7.6 us, dispatch 5.4 us, preempt 11.3 us. The
`kernel` probes measure the same three quantities here, so each implies a
laptop-to-A55 ratio on its own, and whether the three agree is the question.
They are made of different things, wake partly hardware sequencing, dispatch
kernel code, preempt cache and TLB refill, so there is no reason in advance to
expect one number to cover all three.

There is no probe for the A78 and Kryo execution times, because those are drawn
rather than measured and there is no code behind `adc_scan` to run. The
measured A55 column comes from the board campaign, not from here.

Read the kernel probes with care. The i.MX95 numbers were measured natively;
these are measured through Python, whose own overhead is the same order as a
context switch, so they are inflated by the amount the `floor` probe reports.
Use them for the ratio question, not as absolute costs.

Probe groups:

  floor    what the measurement harness costs, so everything else can be read
           net of it
  kernel   wake, context switch, and the refill after being preempted. The
           three with a known A55 value
  compute  raw arithmetic and memory, to separate "this core is faster" from
           "this kernel is faster"
  project  the actual things this thesis quotes: a timing model evaluation, a
           policy forward pass, an allocation, a repair, a short anneal
"""

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the shared modules: timing, tasksets, splits, paths, progress, palette
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out" / "bench"

# measured on the i.MX95, from data/catalogues/hardware_catalogue.json. The only
# real A55 numbers in the project, and therefore the only anchor a conversion
# can hang on.
A55_KERNEL_US = {"wake": 7.6, "dispatch": 5.4, "preempt": 11.3}

NS = 1e-9


def now():
    return time.perf_counter_ns()


class Bench:
    """Run a callable many times and keep the shape of the distribution.

    Median rather than mean, because one scheduler hiccup on a laptop moves a
    mean and does not move a median. The minimum is kept as well: for a pure
    compute probe it is the closest thing to the machine's real capability, with
    every interruption removed.
    """

    def __init__(self, reps, warmup):
        self.reps = reps
        self.warmup = warmup
        self.results = {}

    def run(self, group, name, fn, unit="us", scale=1e-3, setup=None, note="",
            self_timed=False):
        """`self_timed` lets a probe time part of itself.

        Needed whenever the setup for a sample is more expensive than the thing
        being sampled. Evicting a cache costs milliseconds and the refill that
        follows costs microseconds, so timing the pair measures the eviction.
        A self timed probe returns its own duration in nanoseconds and the
        harness records that instead of the wall time around the call.
        """
        arg = setup() if setup else None
        for _ in range(self.warmup):
            fn(arg)
        samples = []
        for _ in range(self.reps):
            if self_timed:
                samples.append(fn(arg))
            else:
                t0 = now()
                fn(arg)
                samples.append(now() - t0)
        samples.sort()
        n = len(samples)
        rec = {
            "group": group, "unit": unit, "n": n,
            "min": samples[0] * scale,
            "p25": samples[n // 4] * scale,
            "median": samples[n // 2] * scale,
            "p75": samples[3 * n // 4] * scale,
            "p90": samples[min(n - 1, int(0.9 * n))] * scale,
            "max": samples[-1] * scale,
            "note": note,
        }
        # spread relative to the median. A probe with a wide spread is measuring
        # the operating system as much as the processor and should not be used
        # as a conversion anchor
        rec["iqr_over_median"] = ((rec["p75"] - rec["p25"]) / rec["median"]
                                  if rec["median"] else 0.0)
        self.results[name] = rec
        print(f"  {name:<24}{rec['median']:>11.3f} {unit:<4}"
              f"  min {rec['min']:>9.3f}  spread {rec['iqr_over_median']:>5.2f}"
              + (f"   {note}" if note else ""))
        return rec


# --------------------------------------------------------------------------
# floor: what the harness itself costs
# --------------------------------------------------------------------------
def group_floor(b):
    print("\nfloor  what measuring costs, subtract this from everything else")
    b.run("floor", "empty_call", lambda _: None,
          note="one python call, the lower bound on any probe here")

    def two_timer(_):
        t = now()
        return now() - t
    b.run("floor", "perf_counter", two_timer,
          note="a clock read, which every sample pays twice")

    ev = threading.Event()

    def set_clear(_):
        ev.set()
        ev.clear()
    b.run("floor", "event_set_clear", set_clear,
          note="an Event touched with nobody waiting, so no switch happens")


# --------------------------------------------------------------------------
# kernel: the three with a known A55 value
# --------------------------------------------------------------------------
def group_kernel(b):
    print("\nkernel  the three costs measured on the i.MX95. The conversion anchors")

    # wake: a thread parked on a condition variable, signalled, running again.
    # The A55 equivalent is the cost of taking a task out of a blocked state.
    def wake_setup():
        st = {"go": threading.Event(), "done": threading.Event(), "run": True}

        def worker():
            while st["run"]:
                st["go"].wait()
                st["go"].clear()
                if not st["run"]:
                    return
                st["done"].set()
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        st["thread"] = t
        return st

    def wake(st):
        st["go"].set()
        st["done"].wait()
        st["done"].clear()

    st = wake_setup()
    b.run("kernel", "wake", wake, setup=lambda: st,
          note="park a thread, signal it, wait for it to run")
    st["run"] = False
    st["go"].set()

    # dispatch: a switch between two runnable threads, with no blocking state to
    # leave. Ping-pong and halve, so one sample is one switch.
    def pp_setup():
        st = {"a": threading.Event(), "b": threading.Event(), "run": True}

        def worker():
            while st["run"]:
                st["a"].wait()
                st["a"].clear()
                st["b"].set()
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        return st

    def pingpong(st):
        st["a"].set()
        st["b"].wait()
        st["b"].clear()

    st2 = pp_setup()
    b.run("kernel", "dispatch", pingpong, setup=lambda: st2, scale=5e-4,
          note="one half of a ping pong, so one context switch")
    st2["run"] = False
    st2["a"].set()

    # preempt: what it costs to come back to a working set somebody else has
    # evicted. Walk an array that fits in cache, have another thread flush the
    # cache by walking a much larger one, then walk the first again.
    try:
        import numpy as np
    except ImportError:
        print("  preempt                 skipped, needs numpy")
        return

    # Only the refill is timed. Evicting a cache costs milliseconds and the
    # refill costs microseconds, so timing the pair would report the eviction
    # and call it a preemption. That is what the first version of this probe
    # did, and it reported 3 ms against the A55's 11 us.
    arrays = {"hot": np.ones(256 * 1024 // 8, dtype=np.float64),
              "cold": np.ones(48 * 1024 * 1024 // 8, dtype=np.float64)}

    def refill(st):
        st["hot"].sum()             # bring the working set in
        st["cold"][::8].sum()       # somebody else runs and takes the cache
        t0 = now()
        st["hot"].sum()             # come back to it. This alone is the cost
        return now() - t0

    def warm_only(st):
        st["hot"].sum()
        t0 = now()
        st["hot"].sum()             # the same touch with the cache still warm
        return now() - t0

    cold = b.run("kernel", "preempt_cold", refill, setup=lambda: arrays,
                 self_timed=True,
                 note="re-touch a 256 KB working set after it was evicted")
    warm = b.run("kernel", "preempt_warm", warm_only, setup=lambda: arrays,
                 self_timed=True,
                 note="the same touch with the cache still warm, as baseline")
    delta = cold["median"] - warm["median"]
    b.results["preempt"] = {
        "group": "kernel", "unit": "us", "n": cold["n"],
        "median": delta, "min": max(cold["min"] - warm["min"], 0.0),
        "p25": delta, "p75": delta, "p90": delta, "max": delta,
        "iqr_over_median": 0.0,
        "note": "cold minus warm: the refill alone. The weakest of the three "
                "anchors, see the note the script prints",
    }
    print(f"  {'preempt':<24}{delta:>11.3f} us  "
          f"  (cold minus warm, the refill alone)")


# --------------------------------------------------------------------------
# compute: raw capability, to tell a faster core from a faster kernel
# --------------------------------------------------------------------------
def group_compute(b):
    try:
        import numpy as np
    except ImportError:
        print("\ncompute  skipped, needs numpy")
        return
    print("\ncompute  raw arithmetic and memory")

    a = np.random.rand(256, 256).astype(np.float32)
    c = np.random.rand(256, 256).astype(np.float32)
    b.run("compute", "matmul_256_f32", lambda _: a @ c,
          note="SIMD float throughput, 33.5 MFLOP")

    v = np.random.rand(1 << 20).astype(np.float64)
    b.run("compute", "vector_sum_8MB", lambda _: v.sum(),
          note="streaming memory, 8 MB read")

    idx = np.random.randint(0, v.size, 1 << 16)
    b.run("compute", "gather_64k", lambda _: v[idx].sum(),
          note="random access, cache miss bound")

    def scalar_loop(_):
        t = 0
        for i in range(2000):
            t += i * i % 7
        return t
    b.run("compute", "python_loop_2k", scalar_loop,
          note="the interpreter, not the processor. A C rewrite removes this")


# --------------------------------------------------------------------------
# project: the numbers this thesis actually quotes
# --------------------------------------------------------------------------
def load_instances():
    """One small, one middling and one large instance, for the project probes."""
    import tasksets
    from timing import Model

    PL = ROOT / "data" / "platforms"
    want = [("small", 1), ("medium", 9), ("large", 20)]
    out = {}
    for label, pid in want:
        ids = tasksets.select(platforms=[pid], per_cell=1)
        if not ids:
            continue
        tid = ids[len(ids) // 2]
        ts = next(t for _, t in tasksets.stream([tid]))
        p = json.loads((PL / f"platform_{pid}.json").read_text())
        m = Model(ts, p)
        m.taskset_id = tid
        out[label] = (tid, ts, p, m)
    return out


def group_project(b, insts):
    if not insts:
        print("\nproject  skipped, no dataset found")
        return
    print("\nproject  what this thesis quotes. Same code that produced the results")

    from allocators import get, repair as repair_with
    from allocators.sa import anneal
    import random as _r

    for label, (tid, ts, p, m) in insts.items():
        n, mcores = len(m.task), len(m.core)
        tag = f"{label}({n}t,{mcores}c)"
        alloc = get("bfd")(m, p)

        b.run("project", f"evaluate_{label}", lambda _: m.evaluate(alloc),
              note=f"{tag} one timing model call")
        b.run("project", f"alloc_bfd_{label}", lambda _: get("bfd")(m, p),
              note=f"{tag} one pass packer")
        try:
            b.run("project", f"alloc_rl_{label}", lambda _: get("rl")(m, p),
                  note=f"{tag} policy forward pass, whole allocation")
        except SystemExit:
            print("  alloc_rl                skipped, no trained checkpoint")

        # a repair: the live path, with everything but a few tasks pinned
        dead = next(c for c in m.core if m.core[c]["core_class"] != "os"
                    and any(v == c for v in alloc.values()))
        red = {**p, "cores": [c for c in p["cores"] if c["core_id"] != dead]}
        from timing import Model as M2
        mr = M2(ts, red)
        mr.taskset_id = tid
        surv = {t: c for t, c in alloc.items() if c != dead}
        for who in ("bfd", "rl"):
            try:
                b.run("project", f"repair_{who}_{label}",
                      lambda _, w=who: repair_with(w, mr, red, surv),
                      note=f"{tag} repair, {len(alloc) - len(surv)} orphans")
            except SystemExit:
                pass

    # a short search, on the middle instance only. Long enough to be the search
    # and short enough not to dominate the run
    if "medium" in insts:
        _, _, p, m = insts["medium"]
        start = get("bfd")(m, p)
        b.run("project", "anneal_100", lambda _: anneal(
            m, start, iters=100, stall=None, checkpoints=(), rng=_r.Random(0)),
            note="100 evaluations of annealing")


# --------------------------------------------------------------------------
# the anchor question
# --------------------------------------------------------------------------
def anchor(res):
    """Does one conversion factor cover all three measured kernel costs?"""
    have = {k: res[k]["median"] for k in ("wake", "dispatch", "preempt")
            if k in res}
    if len(have) < 2:
        return None
    print(f"\n{'=' * 78}\nsanity check against the three measured i.MX95 costs"
          f"\n{'=' * 78}")
    print("  This is a diagnostic, not a conversion. The catalogue numbers were\n"
          "  measured natively on the board. These are measured through Python,\n"
          "  whose threading goes through the interpreter and the GIL before it\n"
          "  reaches the kernel. The two are not the same quantity.\n")
    print(f"  {'quantity':<14}{'here':>11}{'i.MX95':>10}{'here / A55':>13}")
    floor = res.get("event_set_clear", {}).get("median", 0.0)
    ratios = {}
    for k, v in have.items():
        a55 = A55_KERNEL_US[k]
        ratios[k] = v / a55 if a55 > 0 else float("nan")
        print(f"  {k:<14}{v:>9.2f}us{a55:>9.1f}us{ratios[k]:>12.2f}x")

    vals = [v for v in ratios.values() if v == v]
    lo, hi = min(vals), max(vals)
    over = [k for k, v in ratios.items() if v > 1.0]

    # This core is several times faster than an A55. A ratio above 1 says it
    # took longer anyway, which no hardware explanation covers. What is left is
    # the interpreter.
    if over:
        print(f"\n  {len(over)} of {len(vals)} probes come out SLOWER here than "
              f"on the A55.\n  This core is several times the faster of the two, "
              f"so that is not the\n  hardware. It is the interpreter: a python "
              f"Event goes through the GIL\n  before it reaches the kernel, and "
              f"the board numbers were taken in C.")
    if floor:
        print(f"\n  the harness floor alone is {floor:.2f} us, already "
              f"{100 * floor / A55_KERNEL_US['dispatch']:.0f}% of the A55's\n"
              f"  entire dispatch cost. That is the problem in one number.")

    print(f"\n  the three ratios span {lo:.2f}x to {hi:.2f}x, a factor of "
          f"{hi / lo:.2f} apart.")
    if hi / lo < 1.5:
        print("  They cluster, which is worth knowing, but it says the python\n"
              "  overhead is uniform across the three. It does not say the\n"
              "  hardware ratio is any of these numbers.")
    else:
        print("  They do not cluster, so not even the overhead is uniform.")

    print("\n  Nothing here is a conversion factor. The conversion comes from\n"
          "  running this same script on an A55, where both sides then carry\n"
          "  the same python overhead and the ratio is a property of the two\n"
          "  machines rather than of the language.")
    return ratios


def compare(res, other_path):
    """Two profiles, one ratio per probe. This is the real answer."""
    other = json.loads(Path(other_path).read_text())
    them = other.get("probes", {})
    label = other.get("machine", {}).get("label", Path(other_path).stem)
    common = [k for k in res if k in them]
    if not common:
        raise SystemExit(f"{other_path} shares no probes with this run")

    print(f"\n{'=' * 78}\nthis machine against {label}\n{'=' * 78}")
    print(f"  {'probe':<24}{'here':>12}{'there':>12}{'ratio':>9}   group")
    by_group = {}
    for k in common:
        a, c = res[k]["median"], them[k]["median"]
        if a <= 0:
            continue
        r = c / a
        by_group.setdefault(res[k]["group"], []).append(r)
        print(f"  {k:<24}{a:>11.3f} {c:>11.3f}{r:>8.2f}x   {res[k]['group']}")

    print(f"\n  {'group':<12}{'n':>4}{'median ratio':>15}{'min':>9}{'max':>9}")
    for g, rs in by_group.items():
        print(f"  {g:<12}{len(rs):>4}{statistics.median(rs):>14.2f}x"
              f"{min(rs):>8.2f}x{max(rs):>8.2f}x")
    allr = [r for rs in by_group.values() for r in rs]
    print(f"\n  overall median {statistics.median(allr):.2f}x, "
          f"range {min(allr):.2f}x to {max(allr):.2f}x")
    if max(allr) / min(allr) > 2.0:
        print("  the spread is wider than 2x, so a single conversion factor\n"
              "  would misstate some class of work by more than double. Convert\n"
              "  per group, and say which group each quoted number belongs to.")
    return by_group


def machine():
    info = {
        "label": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000,
                    help="timed samples per probe")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--only", nargs="+",
                    choices=["floor", "kernel", "compute", "project"],
                    default=["floor", "kernel", "compute", "project"])
    ap.add_argument("--compare", metavar="JSON",
                    help="another machine's profile. Prints the ratios, which "
                         "is the number you actually want")
    ap.add_argument("--label", help="name for this machine in the output")
    ap.add_argument("--out", default=None,
                    help="where to write the profile. "
                         "Default out/bench/<hostname>.json")
    args = ap.parse_args()

    # one thread everywhere, or the ratio measures how many cores each machine
    # has rather than how fast one of them is
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    info = machine()
    if args.label:
        info["label"] = args.label
    print(f"{info['label']}   {info['processor']}")
    print(f"{info['platform']}")
    print(f"python {info['python']}, {info['cpu_count']} cpus, "
          f"{args.reps} reps per probe, single threaded")
    print("\nclose everything else and put the machine on mains power. A probe "
          "at this\nresolution measures the whole machine, not just this "
          "process.")

    b = Bench(args.reps, args.warmup)
    t0 = time.time()
    if "floor" in args.only:
        group_floor(b)
    if "kernel" in args.only:
        group_kernel(b)
    if "compute" in args.only:
        group_compute(b)
    if "project" in args.only:
        # the project probes are far slower than the rest, so they get fewer
        # samples. A thousand annealing runs is not a benchmark, it is an
        # afternoon
        heavy = Bench(max(args.reps // 20, 30), min(args.warmup, 10))
        heavy.results = b.results
        group_project(heavy, load_instances())

    # Written before anything that can fail. The probes take minutes and a bad
    # --compare path is no reason to throw them away, which is exactly what an
    # earlier version did.
    out = Path(args.out) if args.out else OUT / f"{info['label']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"machine": info, "reps": args.reps,
         "a55_kernel_us": A55_KERNEL_US, "probes": b.results}, indent=1) + "\n")
    print(f"\nwrote {out}   ({time.time() - t0:.0f}s)")

    anchor(b.results)
    if args.compare:
        p = Path(args.compare)
        if not p.exists():
            have = sorted(q.name for q in out.parent.glob("*.json")
                          if q != out)
            print(f"\nno {p}. This run is saved either way, as {out.name}.")
            print("  --compare wants a profile from the *other* machine, "
                  "produced by\n  running this same script there.")
            print("  other profiles on hand: "
                  + (", ".join(have) if have else "none yet"))
        else:
            compare(b.results, p)
    else:
        print("\nto get the conversion factor, run this unchanged on the A55, "
              "bring its\nprofile back, and then:\n"
              f"  python scripts/analysis/bench_cpu.py --compare out/bench/<that>.json")


if __name__ == "__main__":
    main()
