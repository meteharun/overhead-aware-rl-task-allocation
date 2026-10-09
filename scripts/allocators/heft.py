"""HEFT, adapted to partitioned fixed priority.

Heterogeneous Earliest Finish Time (Topcuoglu, Hariri and Wu, 2002), the
standard list heuristic for a dependent task graph on processors of different
speed. Both halves carry over.

Ordering, by upward rank: the longest path from a task to a sink, with
execution averaged over the CPU types and communication over the device pairs,
which is HEFT's own answer to ranking tasks before any placement exists. Upward
rank falls along every edge, so taking the highest ranked ready task is already
a topological order.

Placement, by earliest finish time: periodic preemptible tasks have no finish
time, so the analogue is the response time under the same analysis `timing.py`
uses. Every core competes, with the crossings this placement would create
already charged, and the lowest wins.

The admission test, the os core reservation and the out-of-room fallback are
unchanged from the packers, so the comparison is about the placement rule.
"""

from .common import allocate as _allocate


def upward_rank(model):
    """HEFT's task priority. Longest path from each task to a sink.

    Execution is the mean over the CPU types on this platform, communication
    the mean link delay over the device pairs. On a single device platform the
    communication term is zero and the rank is a longest path over execution.
    """
    cpus = sorted({c["cpu_type"] for c in model.core.values()})
    devs = sorted(model.cores_of_device)
    pairs = [(a, b) for a in devs for b in devs if a != b]
    mean_delay = 0.0
    if pairs:
        mean_delay = sum(sum(lk["delay_us"] for lk in model.route(a, b))
                         for a, b in pairs) / len(pairs)

    def w(tid):
        code = model.task[tid]["code"]
        return sum(model.exec_us[code][c] for c in cpus) / len(cpus)

    memo = {}

    def rank(tid):
        if tid not in memo:
            memo[tid] = w(tid) + max(
                (mean_delay + rank(s) for s in model.adj.get(tid, ())), default=0.0)
        return memo[tid]

    return {tid: rank(tid) for tid in model.task}


def allocate(model, platform):
    rank = upward_rank(model)
    # highest rank first among the ready tasks, ties by id for reproducibility
    return _allocate(model, platform, rule="eft",
                     order_key=lambda t: (-rank[t], t))
