"""Simulated annealing, layered on top of any other allocator.

A wrapper rather than a fourth heuristic: give it a starting allocation and it
walks away from it, keeping the best thing it finds. So `sa_bfd` is best-fit
followed by annealing and `sa_rl` is the policy followed by annealing. Every
annealed variant gets the same move set, schedule and budget, so the starting
point is the only difference between them.

The objective is `penalty.total`, which the thesis minimises, not the
`penalty_hard` the analysis compares on. A single HARD miss is worth 100 to
200 while the whole preference term tops out near 30, so misses dominate and
preference only gives the walk a gradient on plateaus where nothing misses.

Two moves:

  move   one task goes to a different core, anywhere on the platform
  swap   two tasks on different cores trade places, which keeps both core
         loads roughly intact and crosses walls pure moves cannot

Both are biased towards tasks that are currently missing their deadline, since
a uniformly random move on a sixty task set usually touches something that was
already fine.

os cores are legal targets here, unlike in the greedy packers, because the
objective already charges for the receive interference and the preference.
"""

import random
import time
import zlib
from math import exp

ITERS = 4000          # evaluations per instance
SWAP_PROB = 0.35      # share of proposals that are swaps rather than moves
FOCUS_PROB = 0.60     # share of proposals aimed at a task that is missing
T_START_FRAC = 0.08   # opening temperature, as a share of the starting cost
T_END_FRAC = 0.0005   # closing temperature, same units
STALL = None          # evaluations without a new best before giving up, None to run on

# Where the best so far is written down on the way past, so one run answers
# "what would this allocator have produced with a budget of B" for every B in
# the list and the budget curve falls out of the main run.
CHECKPOINTS = (10, 25, 50, 100, 250, 500, 1000, 2000, 4000)


def cost(res):
    """The objective, in one place so every annealed variant minimises it."""
    return res["penalty"]["total"]


def hard(res):
    """Misses plus budget overruns. What the comparison is made on, so it is
    recorded rather than optimised."""
    p = res["penalty"]
    return p["misses"] + p["budgets"]


def seed_for(model):
    """A repeatable seed derived from the instance itself.

    The allocator signature is (model, platform) with nowhere to pass a seed,
    and a fixed seed would make every instance walk the same way. Collisions
    between two task sets do not matter: this picks a stream, never a result.
    """
    key = f"{'|'.join(sorted(model.task))}#{'|'.join(sorted(model.core))}#{len(model.edges)}"
    return zlib.crc32(key.encode())


def anneal(model, alloc, iters=ITERS, rng=None, stall=STALL,
           checkpoints=CHECKPOINTS, track=None):
    """Anneal from `alloc`. Returns (best allocation, stats).

    `iters` counts evaluations of the timing model, not loop passes. A no-op
    proposal, moving a task to the core it is already on, is not charged, so a
    budget of 4000 means 4000 real evaluations and the checkpoints line up with
    the budget they claim.

    `stall` stops early after that many evaluations with no new best. Off by
    default, because a checkpoint past the stopping point would repeat the one
    before it and flatten the budget curve for a reason unrelated to budget.

    `track` is an optional function of the best allocation so far, merged into
    each checkpoint. The core failure experiment uses it to count migrations at
    every budget, which needs the allocation and not just its score.
    """
    rng = rng or random.Random(0)
    tasks = list(model.task)
    cores = list(model.core)
    marks = sorted(m for m in (checkpoints or ()) if m <= iters)

    t0 = time.perf_counter()
    cur = dict(alloc)
    res = model.evaluate(cur)
    cur_cost = start_cost = cost(res)
    cur_missed = [t for t, r in res["tasks"].items() if r["miss"]]
    best, best_cost = dict(cur), cur_cost
    best_hard = start_hard = hard(res)

    def snapshot():
        s = {"hard": round(best_hard, 4), "total": round(best_cost, 4),
             "perfect": best_hard <= 1e-9,
             "ms": round(1000 * (time.perf_counter() - t0), 3)}
        return {**s, **track(best)} if track else s

    # the budget zero point: what the base allocator handed over
    curve = {"0": snapshot()}

    # penalty spans zero to several hundred depending on the platform, so a
    # fixed temperature would be too hot on one instance and frozen on the next
    scale = max(start_cost, 1.0)
    t_hi, t_lo = T_START_FRAC * scale, T_END_FRAC * scale
    decay = (t_lo / t_hi) ** (1.0 / max(iters - 1, 1))
    temp = t_hi

    evals = accepted = proposed = 0
    since_best = 0
    # a degenerate proposal is cheap but not free, and on a one core platform
    # every proposal is degenerate, so the walk needs a way out
    attempts, ceiling = 0, iters * 20

    while evals < iters and attempts < ceiling:
        attempts += 1
        cand = dict(cur)

        pool = cur_missed if (cur_missed and rng.random() < FOCUS_PROB) else tasks
        a = rng.choice(pool)

        if rng.random() < SWAP_PROB and len(tasks) > 1:
            b = rng.choice(tasks)
            if a == b or cand[a] == cand[b]:
                continue
            cand[a], cand[b] = cand[b], cand[a]
        else:
            c = rng.choice(cores)
            if c == cand[a]:
                continue
            cand[a] = c

        temp *= decay
        proposed += 1
        cand_res = model.evaluate(cand)
        cand_cost = cost(cand_res)
        evals += 1

        delta = cand_cost - cur_cost
        if delta <= 0 or rng.random() < exp(-delta / temp):
            cur, res, cur_cost = cand, cand_res, cand_cost
            cur_missed = [t for t, r in res["tasks"].items() if r["miss"]]
            accepted += 1

        if cur_cost < best_cost - 1e-12:
            # `res` always belongs to `cur`, so the best is never scored twice
            best, best_cost, best_hard = dict(cur), cur_cost, hard(res)
            since_best = 0
        else:
            since_best += 1
            if stall is not None and since_best >= stall:
                break

        if marks and evals == marks[0]:
            curve[str(marks.pop(0))] = snapshot()

    # checkpoints the walk never reached, having stalled or run out of
    # proposals, hold wherever it finished
    for m in marks:
        curve[str(m)] = snapshot()

    return best, {
        "sa_start_penalty": round(start_cost, 4),
        "sa_start_hard": round(start_hard, 4),
        "sa_best_penalty": round(best_cost, 4),
        "sa_best_hard": round(best_hard, 4),
        "sa_improvement": round(start_cost - best_cost, 4),
        "sa_evals": evals,
        "sa_proposed": proposed,
        "sa_accept_rate": round(accepted / proposed, 4) if proposed else 0.0,
        "sa_ms": round(1000 * (time.perf_counter() - t0), 3),
        "sa_curve": curve,
    }


def ladder(iters):
    """Checkpoints for a budget of `iters`, extending CHECKPOINTS if bigger.

    The fixed list stops at 4,000, so a longer run would record nothing past
    it. This continues the doubling pattern and always ends on `iters`, so the
    final result is a checkpoint like any other.
    """
    marks = [m for m in CHECKPOINTS if m <= iters]
    m = (marks[-1] if marks else 1000) * 2
    while m < iters:
        marks.append(m)
        m *= 2
    if iters not in marks:
        marks.append(iters)
    return tuple(marks)


def wrap(base, name, iters=ITERS):
    """Turn an allocator into its annealed version.

    Stats from the last call are left on the returned function as `.last`,
    which is how `run_allocators` records them. A side channel, because the
    allocator signature has no second return value, and only valid immediately
    after the call.
    """
    marks = ladder(iters)

    def allocate(model, platform):
        start = base(model, platform)
        best, stats = anneal(model, start, iters=iters, checkpoints=marks,
                             rng=random.Random(seed_for(model)))
        allocate.last = stats
        return best

    allocate.last = None
    allocate.__name__ = name
    allocate.__doc__ = f"simulated annealing started from {name.removeprefix('sa_')}"
    return allocate
