# Overhead-Aware Reinforcement Learning for Real-Time Task Allocation on Heterogeneous Multicore Platforms

Master's thesis, University of L'Aquila, EDISS. Mete Harun Akçay.

This repository holds the timing model, the learned allocator, the benchmark and
every experiment behind `Thesis_MeteHarunAkcay.pdf`.

---

## What the problem is

A STATCOM control platform stabilises voltage on an electrical grid. Its software
is a set of real-time tasks: measurements come in, get conditioned, protection
checks run, a control loop computes outputs, an actuator commits them, and
alongside that sit supervisory logic, communication, telemetry, fault recording
and analytics. That software runs on one board or on several wired together, and
modern industrial SoCs are heterogeneous — an i.MX95 has six A55 cores, a
QCS6490 has four A55 and four A78, an IQ-9100 has eight Kryo cores. Some cores
are isolated for real-time work, some are shared with the operating system.

**Somebody has to decide which task runs on which core of which device.** Today
an engineer does it by hand, and redoes it whenever the workload or the hardware
changes.

Two things make it harder than bin packing:

- **Tasks depend on each other.** Data flows along chains, and a chain has an
  end-to-end budget that no single task owns.
- **Crossing a link is expensive.** Two tasks on one chip talk for free. On
  different devices a message costs a delay plus CPU time at both ends, charged
  to the OS core.

So the cost of a decision is not known until later decisions are made. Placing
task A changes what placing task B costs.

## What the thesis argues

An allocator that learns against a timing model which prices overheads and
communication produces better allocations in one pass than communication-aware
greedy packers, and its advantage survives being given the same search budget as
those packers.

The second half carries the work. Beating a greedy heuristic is easy. Still being
ahead after everyone runs a search is not, because search usually washes out the
starting point. Here it does not.

---

## Repository layout

```
board/          measurement code for the i.MX95, and the campaign it produced
data/
  catalogues/   task catalogue, execution table, hardware catalogue, A55 measurements
  platforms/    platform_1.json .. platform_20.json
  tasksets/     tasksets.jsonl + index.json (16,000 instances) and the three split files
  allocations/  one jsonl per allocator, 6,400 evaluation rows each
  models/       rl.pt (best validation), rl_bc.pt (imitation), rl_last.pt
out/
  writing/      the thesis figures
  runs/         raw experiment output
  reports/      text analyses
  slides/ videos/ platforms/ tasksets/ allocations/   presentation and per-instance figures
reference/      the company STATCOM material and the original brief
scripts/        everything below
```

`data/tasksets/tasksets.jsonl` is 88 MB and is the one file nothing should parse
directly — `scripts/core/tasksets.py` owns the format.

### scripts/

| folder | what is in it |
|---|---|
| `core/` | `timing.py` (the model), `tasksets.py` (the store), `splits.py`, `paths.py`, `progress.py`, `palette.py`, `figstyle.py` |
| `allocators/` | every allocator plus the registry in `__init__.py` |
| `generator/` | platforms, execution table, task sets, splits |
| `rl/` | policy, environment, rollout, training |
| `run/` | the experiment entry points |
| `analysis/` | the tables and the sanity check |
| `fig/` | thesis figures and `generate_figs.py` |
| `visualization/` | one task set, one platform, one allocation |
| `slides/` | the presentation animations |

Every script sits one level deep and starts with the same two lines, which is how
`import timing` works from anywhere:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
```

---

## Running it

Build the dataset:

```
python scripts/generator/gen_platform.py                    20 platforms
python scripts/generator/map_measured_to_catalogue.py       board campaign -> A55 column
python scripts/generator/gen_exec_table.py                  execution table
python scripts/generator/gen_dataset.py --fresh             16,000 task sets
python scripts/generator/split_dataset.py                   train / eval
```

Train and evaluate:

```
python scripts/rl/train.py
python scripts/run/run_allocators.py --eval
python scripts/analysis/analyse_allocations.py
python scripts/run/repair.py
python scripts/run/repair_search.py
python scripts/fig/generate_figs.py all
```

Two of those refuse to run twice on purpose. `gen_exec_table.py` will not
overwrite an existing execution table without `--force`, because every allocation
on disk was computed against the one that is there. Move `data/models` aside
before retraining: `allocators/learned.py` registers whatever checkpoint it
finds, so a stale one silently joins the results table.

### Search budget

`--sa-evals N` sets the candidate evaluations every annealed allocator gets.
The default of 4,000 is what the thesis compares starting points at. A much
larger budget asks the other question — whether an instance is solvable at all:

```
python scripts/run/run_allocators.py --only sa_rl --eval --sa-evals 500000
```

It is the same search either way, so it is a setting rather than a separate
script. The checkpoint ladder extends with the budget, so the earlier points stay
where they were and new ones appear above them.

### Figures

`scripts/fig/generate_figs.py` numbers every figure the way the thesis does:

```
python scripts/fig/generate_figs.py            list them
python scripts/fig/generate_figs.py 1 3 5      just those
python scripts/fig/generate_figs.py 13-20      a range
python scripts/fig/generate_figs.py all --collect
```

Sixteen of the twenty are built by a script. Figure 1 is exported by hand from
`architecture.drawio` and figures 7 to 9 are TikZ drawn in the document.

**Figure conventions.** In the platform and allocation figures, hue is CPU type
and lightness is core class, lightest being the OS core. In the result charts,
`core/figstyle.py` holds the registry: deep blue is the policy, amber best fit,
violet worst fit, teal earliest finish time, and a criticality-ordered variant is
the same hue one step darker and dashed. In the schedule figures, hatching is
kernel overhead and black is execution past a deadline.

---

## The timing model

`scripts/core/timing.py` turns a complete allocation into response times, chain
latency, per-core loads and a penalty. Everything else depends on it, and the
learned policy's reward is this model rather than a proxy, so when the model
changed the objective followed with no change to the learner.

**Response time** is a fixed point, iterated per core in deadline-monotonic order:

```
R = (wake + C_eff + Σ ceil(R/P_h)·(dispatch + C_h) + Σ (ceil(R/P_h)−1)·preempt) × tail
```

The iteration stops when it converges or when it passes the deadline, whichever
comes first. The cap of 1,000 updates was never reached: the observed maximum
across 720,606 computations is 15.

**Effective execution** adds the CPU cost of sending each outgoing cross-device
message to the producer. Receive and forwarding work lands on the destination's
OS core at interrupt priority.

**Tail factors** are measured p99.9 / p50 of execution time per core class:
isolated 1.1, general 1.2, OS 1.3. Execution speed does not change with core
class, only the tail does, which is why execution is keyed on
`(code, cpu_type)` and the tail on `core_class`. The factor is applied once, to
the whole response window.

**Chains are sampled.** A consumer reads the latest value and never waits, so
each hop costs a link delay, one consumer period of sampling wait and the
consumer's own response time. Latency is the worst over all paths between the
budget endpoints.

**Penalty** is deadline misses (internal weight × external weight) plus chain
overruns plus a core-class preference tiebreaker. Reported quality uses only the
first two.

**Nothing derived is stored.** Criticality, external weight and preference come
from the period. Priority is deadline-monotonic per core. Whether an edge crosses
a link is computed from the allocation.

---

## The allocators

| name | what it does |
|---|---|
| `bfd`, `wfd` | best-fit and worst-fit packers, communication-aware |
| `heft` | earliest finish time, adapted to periodic fixed-priority |
| `*_crit` | the same three, taking the most critical ready task first |
| `rl` | the learned policy, one forward pass per placement |
| `sa_*` | any of the above followed by simulated annealing |
| `exact` | branch and bound, used only to produce feasibility witnesses |

`sa.py` is a wrapper, not a seventh heuristic: every annealed variant gets the
same move set, schedule and budget, so the only thing that differs between
`sa_bfd` and `sa_rl` is where the search started. That is the comparison the
thesis is built on.

**The admission test is the scorer, exactly.** `allocators/common.py`
reimplements the response-time equation so a packer can refuse a core without
evaluating the whole allocation. A cheaper approximation would not be an
admission test — it would just move the miss from the allocator to the report.

**Nothing in the policy is keyed to identity.** No device id, no platform id, no
CPU-type or link-type category. A core is its numbers: what this task would cost
on it, its tail, its kernel charges, how loaded it already is, how far it sits
from the task's neighbours. An unseen platform is different numbers in the same
slots, which is why it transfers.

---

## The dataset

16,000 task sets: 20 platforms × 4 coupling levels × 5 load levels × 40
instances. Each graph is a main chain plus side branches plus independent tasks,
with sampled edges.

**Load** targets the *measured* effective load after a reference allocation, not
an estimate, because a fixed target meant an easy instance on a small platform
and an impossible one on a large platform. The generator grows the task count
until the measured load lands within 0.02 of 0.6 / 0.7 / 0.8 / 0.9 / 1.0. Every
level hits its target: mean error is at most 0.002, and level *k*'s p95 sits below
level *k+1*'s p5.

**Coupling** moves two things at once, the share of independent tasks and the
number of extra edges beyond the tree, because removing independent tasks alone
left the edge count flat at about one per task.

**Budgets** are the chain's own latency floor × 1.8, computed from the task graph
alone, so the same workload is judged the same way whatever it runs on.

**Splits.** 80% of each cell on platforms 1–15 trains, 20% is workload
evaluation, and platforms 16–20 are held out entirely. After reserving 200 for
validation: 9,400 train, 200 validation, 2,400 workload eval, 4,000 hardware
eval.

---

## Results

All on the same 6,400 held-out task sets. Timing penalty excludes the preference
tiebreaker; perfect means no deadline miss and no chain overrun.

**One pass, no search:**

| allocator | perfect | mean penalty | median |
|---|---|---|---|
| BFD | 19% | 343.4 | 294.4 |
| WFD | 19% | 322.8 | 252.0 |
| HEFT | 25% | 404.6 | — |
| heuristic portfolio | 41% | 141.2 | — |
| **RL** | 20% | **110.3** | **14.9** |

RL is 66% below the strongest individual heuristic in mean penalty, at 8.3 ms per
instance against 47.4 ms for running all six heuristics. It is level on perfect
rate, which is a real limitation: it minimises total penalty, not how often it
lands exactly on zero.

**After 4,000 candidate evaluations of annealing:**

| starting point | perfect | mean penalty |
|---|---|---|
| BFD | 71% | 6.3 |
| WFD | 72% | 7.0 |
| HEFT | 68% | 11.5 |
| **RL** | **76%** | **4.6** |

Those runs take between 1.561 and 1.582 seconds each, so the difference is not
bought with time. The learned start also reaches a given quality on far less
search: no heuristic matches one-pass RL below about 74 evaluations, and RL has
already reached 38.4 at the 100-evaluation checkpoint where the best heuristic
is at 97.6.

**Transfer.** On 2,400 unseen workloads from training platforms SA–RL reaches
3.7, and on 4,000 instances from the five held-out platforms it reaches 5.1 — a
factor of 1.38, the smallest increase of the seven methods. The heuristics also
do worse on the held-out pool despite having no training set, so the increase
cannot be attributed to transfer alone.

**After a core failure**, over 12,800 single-core failures, repair beats rebuild
for every allocator at every matched search budget. With ten candidate
evaluations RL repair reaches 90.1 in 8.12 ms against 139.4 in 11.8 ms for RL
rebuild; at 1,000 evaluations, 21.5 against 27.4.

**How much is solvable at all.** Taking the best allocation any method produced,
82.8% of the 6,400 instances have a zero-penalty witness at the 4,000-evaluation
budget. Continuing the search on the ones that were still unsolved raises that to
96.2%. No instance in the dataset is infeasible on capacity: bounding every task
by its cheapest possible core placement gives a maximum required utilisation of
0.930.

---

## Where the numbers come from

| | source |
|---|---|
| A55 execution times | measured on a Verdin i.MX95, `board/` |
| core-class tails 1.1 / 1.2 / 1.3 | measured, p99.9 / p50 per class |
| A55 wake / dispatch / resume | measured, 7.6 / 5.4 / 11.3 µs |
| A78 and Kryo overheads | scaled from A55, not invented separately |
| other CPU types' execution times | drawn uniformly in 0.10–0.30 of the period, seed 20260820 |
| link delay and CPU cost | assumed |
| device core counts and CPU mixes | real parts |
| periods, deadline range, weights | fixed by the brief |
| preference penalty table | chosen |
| budget tightness 1.8 | tightened until the term binds |
| load and coupling levels | calibrated against measured allocator behaviour |

**How the CPU overheads were scaled.** The three costs are made of different
things and a bigger core does not speed them up equally. `dispatch` is almost all
scheduler code plus a register save and restore, so it tracks integer throughput
and a wide out-of-order core cuts it most. `wake` adds exiting an idle state,
part of which is hardware sequencing that does not care how wide the pipeline is.
`preempt` is dominated by refilling cache and TLB after someone else ran, and a
bigger core has more state to reload, so it improves least. Within every CPU
`preempt > wake > dispatch` holds, which is what the mechanism says it should.

---

## Design decisions

**All edges are sampled.** In the company's own STATCOM graph it is 15 sampled
against 2 triggered, which is typical for fixed-interval control. Periods no
longer propagate along the graph and chain latency loses its branch.

**Star topology with relaying.** A star puts every device one hop from the centre
but two leaves have no direct link, and an allocator will happily put connected
tasks on two different leaves. A message between them is relayed: the sender pays
the first hop, every device it lands on pays a receive, and a relaying device also
pays the send of the next hop.

**Period belongs to the code, not the task.** A protection check does not run at
2 ms.

**Execution times are p50 on a quiet core**, not a tail percentile, because the
model multiplies the whole response window by the core-class tail factor. Taking
a tail percentile for execution as well would charge the tail twice.

**One extra pair feature once the chain budget started binding.** At tightness
1.8 the budget term stops being decoration, and the policy was being punished for
something it could not see. `chain_headroom` is the share of the budget still
unspent if the task goes on this core.

### Tried and rejected

- **A last resort that minimises total added load.** Much worse: it stacks tasks
  onto already-full cores rather than using an idle core elsewhere. The last
  resort spreads.
- **Extra-edge fractions of 0.7 and 1.2.** The coupling axis saturates past about
  0.4; above it only produced cells nobody could solve.
- **Budgets from the reference allocator's achieved latency.** The same workload
  got a different budget on different hardware.
- **A preference penalty charged per rank step.** It grew with instance size,
  which made large instances look worse for no reason.
- **Judging cell quality by perfect rate.** Wrong metric. Cells where nobody is
  perfect separate the allocators best; cells where everybody is perfect separate
  nothing. Quality is measured by tie rate.
- **Letting the policy choose the placement order too.** A second head picking
  which ready task to place next. It lost in one pass and tied after annealing,
  so there was no case where it won. The deadline-monotonic order is already a
  good order; the value is in choosing the core. The head is still in
  `rl/policy.py`, unused.
- **Shrinking execution times to make instances fit.** Backwards. The generator
  targets a measured effective load, so smaller tasks mean *more* tasks to reach
  it and each one costs a fixed switch.
- **Lowering the load band to fix multi-device platforms.** Also backwards. Load
  was never the binding constraint there: best fit deliberately fills one core to
  the brim, so the mean is irrelevant and the max decides.

---

## Limitations

The results concern one training run and one timing model with measured,
synthetic and assumed parameters. Execution times for the processor types other
than the A55 are sampled independently, so they do not preserve a fixed speed
ranking between CPU types. Link costs are constant and do not vary with data
size, traffic or queueing. The model needs validation against observed response
times and chain latencies before it can support deployment decisions.

Criticality is derived from period in the benchmark and the generated deadline
bands do not overlap across criticality classes, so deadline-monotonic priority
and criticality order coincide and their effects cannot be separated.

Where a task misses its deadline the response-time iteration stops early, so the
reported value is a lower bound and any chain through that task is marked
inexact. This never changes the feasibility verdict, because such a chain always
contains a task that has already missed, but it understates the penalty of
allocations that are already infeasible.
