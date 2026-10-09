# Thesis Brief

**This is the original specification, written before any code existed.** It is
kept as a record of what the project set out to build. It is not a description of
what was built: the design moved in several places while the work was done, and
where the two disagree the delivered system is authoritative. `README.md` in the
repository root describes the system as it stands, and the last section here
lists every place the two differ.

---

## 1. Files

Four files, all in `reference/`.

| file | what it is |
|---|---|
| `THESIS_BRIEF.md` | this document, the original specification |
| `STATCOM_IMX95_REAL_TASK_WORKLOAD.md` | the real workload from the company: 14 tasks, their dependency chain, what to measure |
| `statcom_workload_runner.c` | the company's reference C implementation of that workload |
| `STAMP Production 2026-07-30.csv` | the company's task inventory |

The two STATCOM files describe the real application. Use them to sanity check
that the model can express a real system. Do not copy numbers out of them.

---

## 2. Title

Working title:

**Overhead-Aware Allocation of Dependent Real-Time Task Graphs onto
Heterogeneous Connected Platforms**

As submitted:

**Overhead-Aware Reinforcement Learning for Real-Time Task Allocation on
Heterogeneous Multicore Platforms**

---

## 3. Problem

Input: a set of real-time tasks that send data to each other, and a description
of the available hardware.

Output: which device and which core every task runs on.

Scheduling inside a core is partitioned fixed-priority. Priorities follow from
the allocation, so the allocation is the only decision.

Two things make this harder than classical task-to-core bin packing:

1. **Tasks depend on each other.** Data flows along chains, and a chain has an
   end-to-end deadline that no single task owns.
2. **Crossing a link is expensive.** Two tasks on the same chip talk for free.
   On different devices it costs a large delay plus CPU time on both ends.

Because of (2) the cost of a decision is not known until later decisions are
made. Placing task A changes what placing task B costs.

---

## 4. Data model

### Task

| field | |
|---|---|
| `task_id` | |
| `code` | which code it runs, looks up execution time by CPU type |
| `period_us` | one of 100, 500, 1000, 2000. Nothing else. |
| `deadline_us` | between 0.8 and 1.0 times the period |
| `internal_weight` | a float between 1.0 and 2.0 |

Five fields. Nothing else.

Criticality, external weight and core preference are **inferred from the
period**, never stored:

| period | criticality | external weight | prefers |
|---|---|---|---|
| 100 | HARD | 100 | isolated, general, os |
| 500 | FIRM | 10 | isolated, general, os |
| 1000 | SOFT | 1 | os, general, isolated |
| 2000 | SOFT | 1 | os, general, isolated |

Cost of a task missing its deadline is `internal_weight × external_weight`. So
100 to 200 for HARD, 10 to 20 for FIRM, 1 to 2 for SOFT. The tiers never
overlap.

Core preference is a **preference, not a rule**. A 100 µs task can go on an os
core if the isolated ones are full. It costs a small penalty. Nothing is ever
forbidden.

Sporadic and event-driven tasks use their minimum inter-arrival time as the
period. There is no activation type field.

Best-effort tasks such as logging and AI inference get period 2000 and a low
internal weight. No separate task type.

**Priority is derived, never stored, and the policy does not choose it.**
Deadline-monotonic on each core: shortest deadline gets highest priority. This
is provably optimal for fixed-priority with deadlines no longer than periods,
so searching over priorities can only do worse.

### Edge

| field | |
|---|---|
| `src`, `dst` | |
| `semantics` | `sampled` or `triggered` |

Three fields. Message size is not modelled.

`sampled`: the consumer reads the latest value on its own schedule. Producer
and consumer may have different periods.

`triggered`: the producer wakes the consumer. A task cannot run more often than
what wakes it, so a triggered task's period equals its producer's period, or
the shortest one if several producers trigger it. Multiple triggers are normal.

Tasks in a chain may have different periods. A 100 µs task feeding a 1 ms task
means the consumer ignores most of what the producer wrote. The other direction
means the consumer reads the same value several times.

### End-to-end budget

| field | |
|---|---|
| `src`, `dst` | first and last task |
| `budget_us` | |

The path between them is found by walking the edges. Never store it.

A chain has a floor the allocator cannot beat. Each sampled hop costs up to one
consumer period of waiting, so the periods along the path set a minimum. A
budget below that floor makes the instance unsolvable. Always check.

### Hardware

**Core**: `core_id`, `device_id`, `cpu_type`, `core_class` (os / general / isolated)

**Device**: `device_id`

**Link**: `dev_a`, `dev_b`, `link_type`

Nothing else. A single-device platform is one device and zero links, and no
code may special-case that.

### Parameter tables

**Execution**, keyed by `(code, cpu_type)`, gives execution time.

A single speed factor per core is **wrong** and must not be used. Two tasks can
rank differently on two CPU types, because an out-of-order core helps a
branch-heavy task far more than a task doing regular arithmetic. Only a full
table is correct.

**Per CPU type** (A55, A78, Kryo, ...)

| | |
|---|---|
| wake | core was idle, timer fires, kernel starts the task |
| dispatch | core was busy, kernel hands over between two tasks |
| preempt | a shorter-period task interrupts a running one, then it resumes |

These belong to the CPU type, not the device. They are the kernel's scheduler
code running on the core doing the switch, so a faster core runs them faster.

**Per core class**

| class | tail multiplier |
|---|---|
| isolated | 1.10 |
| general | 1.21 |
| os | 1.30 |

About interference, not speed, which is why it is keyed on class. These three
are measured values from real hardware, not placeholders.

**Per link type** (usb, eth_1g, eth_10g, ...)

| | |
|---|---|
| delay | how long the message takes to arrive. Uses no CPU. |
| cpu cost | charged to the sending task, and again on the receiving device |

Two numbers. No bandwidth, no per-byte term, no separate sender and receiver
values.

**Communication between two cores on the same chip is free.**

Whether an edge crosses a link is **computed, never given**. It depends on
where both endpoints were placed. This is the most important structural fact in
the design.

---

## 5. Timing model

### Execution time per task

```
C_eff = execution(code, cpu_type)
      + link cpu cost, for every edge of this task that crosses a link
```

Edges staying on one chip add nothing.

For an edge crossing a link, the receiving cost does not go to the consumer
task. It goes to the **os core of the destination device**, even if nothing was
placed there. That is where the network interrupt is handled in reality.

### Response time

On each core, sort by deadline, shortest first.

```
R = ( gap + C_eff(i)
      + sum over higher priority j of ceil(R / period_j) * C_eff(j)
      + k * preempt )
    * tail_multiplier(core_class)
```

Iterate until R stops changing.

`gap` is `wake` for the highest priority task on a core, `dispatch` for the
rest.

`k` is the number of preemptions. **Only a strictly shorter period can
preempt.** Equal-period tasks are released at the same instant, so the higher
priority one runs to completion first. That is blocking, not preemption, and
costs no switch.

The tail multiplier applies to the whole response time, not just execution.

A task misses if `R > deadline`.

### Chain latency

```
total = R(first task)

for each hop from A to B:
    if A and B are on different devices:
        total += link delay
    if edge is sampled:
        total += period(B)      worst case, B just missed the update
    else:
        total += wake           B is woken directly
    total += R(B)
```

Violated if `total > budget_us`.

### Core load

```
core_load = sum over tasks on that core of C_eff / period
            plus any link receiving cost landing on that core
```

### Objective

```
penalty =   sum over missed tasks of  internal_weight × external_weight
          + sum over violated budgets of  weight of the worst task on the path
          + small penalty per task placed off its preferred core class
```

Lower is better. This number is the score and the training reward.

OPEN: the size of the core preference penalty. It must stay far below the cost
of any deadline miss.

---

## 6. Solution

An allocator that places tasks one at a time, learned with reinforcement
learning, scored by the timing model above.

### Why not the alternatives

- **Classical bin packing** cannot see that separating two communicating tasks
  is expensive. It is a baseline, not the answer.
- **Learning from an optimal solver** needs an oracle. Response time is an
  iterative fixed point with chain latency on top, which does not encode
  exactly into a solver, so the oracle would be approximate and the model would
  inherit its ceiling.
- **Predicting the best core for each task independently** ignores that cores
  have finite capacity. Two tasks can want the same core. Decisions must be
  sequential.

### The policy

At each step, score every pair of (unplaced task, core). Take the best pair.
Repeat until everything is placed. So the policy decides **both which task to
place next and where to put it.**

Architecture:

- Encode each task with a shared network, with message passing over the edges
  so a task's representation carries information about its neighbours.
- Encode each core with a shared network: device, class, CPU type, current load.
- One round of attention so tasks see cores and cores see tasks.
- Score each pair from the two encodings plus pair features: predicted slack if
  this task went on this core, resulting core load, and whether this placement
  puts it across a link from an already-placed neighbour.

**No dimension of the network may depend on the number of tasks or cores.** Use
shared per-entity encoders and a shared pair-scoring head. The same weights must
accept 6 cores or 24 cores. Write a test that builds the network for several
core counts and checks the parameter count is identical.

### Reward

Terminal, equal to minus the penalty.

Normalise per instance against a simple heuristic's score on the same instance.
Without this, a small easy problem and a large hard one have reward scales an
order of magnitude apart.

Chain latency is only known when the last task is placed, so per-step rewards
based on partial allocations can mislead. If shaping is used at all, base it on
a lower bound, not a guess.

### Repair

After the policy finishes: find the task contributing most to the penalty, try
moving it, keep the move if the score improves, repeat.

Report policy alone, policy plus repair, and heuristic plus repair as separate
rows. Otherwise it is impossible to tell which produced the result.

---

## 7. Dataset

All synthetic. No measurements.

Quantities marked OPEN were undecided when this was written. All of them
were settled later; `README.md` has the values.

1. **Shape.** A main chain of tasks, plus side branches. A branch is a task that
   reads from the chain but is not part of it, like telemetry.
   OPEN: chain length, number of branches.
2. **Periods.** From {100, 500, 1000, 2000}. Absolute, independent.
   OPEN: how they are drawn.
3. **Code.** From a catalogue, so execution times come from the table.
   OPEN: the catalogue.
4. **Deadlines.** 0.8 to 1.0 of the period. OPEN: how drawn within that range.
5. **Internal weights.** Float 1.0 to 2.0. OPEN: how drawn.
6. **Edge kinds.** `sampled` or `triggered`. OPEN: the mix.
7. **Budgets.** Must be above the chain floor. OPEN: how far above.
8. **Hardware.** Device count, cores per device, class mix, link types.
   OPEN: all of it.
9. **Scale until a target CPU load.** OPEN: the target loads.

Store the generator settings with every instance. They become the x-axis of the
results.

### Two difficulty axes

**Load** decides whether tasks fit on cores. **Budget tightness** decides
whether chains fit their deadlines. They are independent, and the dataset should
be banded on both.

### Splits

Train and evaluation pools with no overlap. OPEN: counts.

Also hold out entire hardware configurations, so the policy is evaluated on a
platform shape it never saw. OPEN: which ones.

---

## 8. Evaluation

### Baselines

- Worst-fit, best-fit and rate-monotonic first-fit packing. These ignore
  communication.
- The same with a communication-aware task ordering.
- A dependency-aware list scheduling heuristic adapted to periodic
  fixed-priority tasks. Document every adaptation, an unfair one is a strawman.
- Simulated annealing with the same evaluator and a fixed budget of evaluator
  calls. This is the strongest competitor and may beat the policy on quality.
  The policy's advantage is speed. Say so honestly.
- A hand-written expert allocation where one exists.

### Main result

Not one aggregate number. Sweep how much traffic crosses links and how deep the
chains are, and report where the policy beats the heuristics and where it ties.
Ties are findings, not failures.

**Multi-device instances are required.** On a single device this model has
little coupling: on-chip communication is free, so placement affects chain
latency only through response times. The interesting decision is which tasks are
worth sending across a link, paying a large delay plus CPU on both ends to gain
a whole device of capacity.

### Statistics

Paired comparisons on the same instances, with a Wilcoxon signed rank test and a
bootstrap confidence interval on the mean difference. A table of means cannot
tell whether two methods differ.

Several training seeds, with the spread reported. One seed is not evidence.
OPEN: how many.

### Expect to be asked

- The reward comes from the same model that judges the answer, so the policy
  will find that model's blind spots.
- Any step that picks the better of two candidates cannot lose by construction.
  Never report such a number as the policy's performance.

---

## 9. Out of scope

Hardware measurement. Thermal and power. Runtime migration. Online learning.
Cache contention between tasks sharing a last-level cache. Link bandwidth and
congestion.

Every exclusion goes in the thesis as a stated assumption with one line of
justification.

---

## 10. Build order

1. Data model, plus one hand-written example instance.
2. Timing model. Run it on the example and show me the tables. Check the numbers
   by hand before continuing, everything downstream trusts this.
3. Sanity check: move one task across a link, confirm the chain breaks and two
   cores get busier. If that does not happen, the evaluator is wrong.
4. Code catalogue, then the generator.
5. Classical baselines.
6. Policy, with the size-agnosticism test.
7. Training loop.
8. Evaluation and statistics.

---

## 11. Where the delivered system differs from this brief

Written after the work was finished. Where this list and the sections above
disagree, this list is right.

### Model

**Edges are all sampled. The `semantics` field is gone.** The brief has
`sampled` or `triggered` as a stored field. In the company's own STATCOM graph
it is 15 sampled against 2 triggered, which is typical for fixed-interval
control, so the triggered case was dropped. Periods no longer propagate along
the graph and the chain-latency calculation lost its branch.

**The response-time recurrence charges more than the brief says.** Section 5
gives `sum ceil(R/P_j) * C_eff(j)` for interference and `wake` only for the
highest-priority task. As implemented, every interfering release also costs a
`dispatch`, because the kernel spends the switch whoever it is switching to, and
the `wake` is charged to every task analysed. Counting only the execution was a
bug, found by checking the model against a simulator. It invalidated every
experiment that had run to that point.

**`preempt` is called `resume` in the thesis.** Same constant, same measurement.

**Core preference changed for FIRM tasks.** The brief has HARD and FIRM both
preferring isolated cores. The implemented table keeps HARD on isolated and puts
FIRM and SOFT on os, so an isolated core is reserved for hard work and anything
else pays for taking the seat. Table 3.4 of the thesis has the values.

**The general-class tail is 1.2, not 1.21.** Measured 1.21, rounded.

**Core load is reported twice.** The brief defines one number, execution over
period. The evaluator reports a raw load matching that, and an effective load
that also carries kernel costs, preemption and the tail.

### Policy

**The policy does not choose which task to place next.** The brief has it score
every (task, core) pair and take the best. The shipped policy walks a fixed
order — topological, or criticality-ordered on single-device platforms — and
chooses only the core. The order head exists in `rl/policy.py` and is disabled:
it lost in one pass and tied after annealing, so there was no case where it won.

**No attention layer.** The brief asks for one round of attention so tasks see
cores and cores see tasks. A plain concatenation of the two embeddings plus the
pair features scored as well with one less thing to tune.

**The reward baseline is the policy's own greedy rollout**, not a heuristic's
score on the same instance. Each instance is run twice, once sampled and once by
argmax, and the policy is rewarded for beating its own deterministic self.

**"Repair" means something else now.** In the brief it is a local improvement
loop run after the policy finishes, which is what simulated annealing became. In
the thesis it is the response to a core failure: pin the survivors, reallocate
the orphans.

### Dataset and evaluation

**Execution times for the A55 are measured, not synthetic.** The brief says "all
synthetic, no measurements", and section 9 puts hardware measurement out of
scope. A measurement campaign on a Verdin i.MX95 supplies the A55 execution
times, the kernel overheads and the core-class tail factors. `board/` holds the
code and the results. The other two CPU types are still drawn.

**The second difficulty axis is coupling, not budget tightness.** The brief bands
the dataset on load and budget tightness. Budget tightness is fixed at 1.8 across
the whole benchmark, and the second axis is how densely the task graph is
connected.

**Rate-monotonic first-fit was not built.** The baselines are best-fit,
worst-fit and HEFT, each with a criticality-ordered variant, plus simulated
annealing over any of them and a branch-and-bound reference. No hand-written
expert allocation exists for these instances.

**Statistics are bootstrap confidence intervals, not Wilcoxon**, and there is
one training seed rather than several. Both are stated as limitations in the
thesis.
