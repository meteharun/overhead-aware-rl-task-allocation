# STATCOM i.MX95 Real Task Workload for CPU Core Allocation Testing

This document defines a realistic workload to replace formula-only CPU-load experiments when testing static task-to-core allocation on i.MX95 for the STATCOM control platform thesis.

## 1. Purpose

- Provide a task graph that looks like a real STATCOM runtime, not a synthetic benchmark.
- Give the thesis owner concrete tasks to feed into the allocator: measurements, control, IEC 61850 communication, IPC, shared memory, telemetry, TFR recording, and AI side load.
- Keep the workload implementable on i.MX95 Linux with PREEMPT_RT using C and POSIX threads.

## 2. Ground Rules

- Target: single i.MX95 device.
- Runtime model: Linux PREEMPT_RT on A55 cores.
- Allocation model: static core affinity plus fixed real-time priorities.
- Scope: on-device workload generation and timing measurement.
- Non-goal: online AI migration or runtime policy mutation.

## 3. Why This Replaces Formula-Only Tests

Formula-based tests usually exercise only arithmetic throughput. They do not expose the main reasons a mapping fails on a real control platform:

- interrupt bursts from measurement or network traffic,
- cache-line bouncing on shared state,
- queue backpressure between measurement, control, and telemetry tasks,
- recorder and log IO starting during a fault event,
- timing interference between real-time and best-effort threads,
- scheduler effects when SCHED_FIFO tasks share or fight over a core.

The workload below is intended to trigger those effects in a controlled way.

## 4. Reference Task Set

The task set is intentionally mapped to the service names already present in the repository, even though those folders are currently placeholders.

| Task ID | Candidate Service / Ownership | Real Runtime Role | Trigger / Period | Scheduling Class | IPC / Shared Data | Why It Matters for Allocation |
|---|---|---|---|---|---|---|
| T01 | io-ingress-service | Sampled-value or field-measurement ingest | 80 us or 100 us periodic; optionally IRQ-driven | SCHED_FIFO high | DMA/ring buffer to measurement snapshot | Represents NIC or field ingress pressure and wakeup jitter |
| T02 | io-ingress-service | Signal conditioning, scaling, plausibility check, anti-alias filter | Same cycle as T01 | SCHED_FIFO high | Single-producer shared snapshot | Exposes cache locality and bounded handoff cost |
| T03 | ctrl-coordination-logic | Fast protection checks: over/under-voltage, bad quality, missing data | 80 us to 1 ms | SCHED_FIFO highest | Read-only snapshot from T02 | Must remain deterministic even under recorder or telemetry load |
| T04 | ctrl-coordination-logic | Inner control loop: dq transform, PLL, current/voltage regulator, modulation update | 100 us or 1 ms | SCHED_FIFO highest | Shared control-state block | Main hard real-time control consumer |
| T05 | sys-orchestrator | Supervisory control: setpoint arbitration, mode transitions, fallback state machine | 1 ms to 10 ms | SCHED_FIFO medium | Message queue plus shared state | Represents state-machine bursts that must not disturb fast loop |
| T06 | sys-orchestrator | IPC broker / ownership checks / sequence validation | Event-driven per update | SCHED_FIFO medium | eventfd plus shared-memory rings | Models realistic software glue overhead, not just math load |
| T07 | io-ingress-service + sys-orchestrator | IEC 61850 SV/GOOSE encode-decode hot path | Event-driven plus publish windows | SCHED_FIFO high | Ring buffer plus socket path | Adds packet burst behavior and network-driven interference |
| T08 | sys-telemetry-service | Runtime telemetry collector: per-core load, queue depth, miss counters, jitter | 10 ms periodic | SCHED_FIFO low or SCHED_OTHER | Read-only snapshots | Required for thesis evidence, but can distort results if placed badly |
| T09 | sys-telemetry-service | TFR event recorder trigger and metadata capture | Event-triggered | SCHED_FIFO low | Circular buffer header write | Critical because it starts exactly when the system is disturbed |
| T10 | sys-telemetry-service | TFR bulk writer / waveform export | 10 ms background flush or event burst | SCHED_OTHER | Circular buffer to file or RAM sink | Produces memory and IO contention during faults |
| T11 | ai-inference-service | Feature extraction window assembly | 10 ms to 100 ms | SCHED_OTHER or low FIFO | Read-only tap from telemetry or measurements | Realistic side workload for predictive maintenance |
| T12 | ai-inference-service | Inference execution and confidence scoring | 100 ms to 1 s | SCHED_OTHER | Feature queue | Shows how non-critical compute competes with control workload |
| T13 | hal-valve-actuator | Output dispatch / actuator command packaging / safety command commit | 100 us to 1 ms | SCHED_FIFO high | Shared command block with seq count | Closes the control loop and adds write-side synchronization |
| T14 | sys-telemetry-service | Logging and evidence packaging | Best effort | SCHED_OTHER | Log queue | Should never steal time from protection or control threads |

## 5. Recommended First Mapping for i.MX95

This is not the final answer. It is the baseline candidate that the allocator should try to beat.

| Core | Primary Role | Recommended Tasks |
|---|---|---|
| Core 0 | Housekeeping / Linux | ptp4l, phc2sys, kernel work, low-priority logging |
| Core 1 | Supervisory control | T05, part of T06, low-rate command handling |
| Core 2 | Fast measurement and protection | T01, T02, T03 |
| Core 3 | Inner control and actuator path | T04, T13 |
| Core 4 | RT communications and evidence trigger | T07, latency-sensitive part of T06, T09 |
| Core 5 | Best-effort analytics and recorder writer | T08, T10, T11, T12, T14 |

Rules for realism:

- T01 to T04 must stay on isolated or near-isolated cores in every serious mapping candidate.
- T09 may run with low overhead near the real-time path, but T10 must not.
- AI and logging tasks must be movable stressors, not fixed on control-critical cores.
- If the thesis tests migration, migration of T01 to T04 should remain effectively zero in the accepted deployment profile.

## 6. Interaction Graph

The realistic execution chain is:

1. T01 acquires new measurements.
2. T02 conditions and validates them.
3. T03 performs protection checks.
4. T04 computes control outputs.
5. T13 commits actuator command state.
6. T05 updates supervisory state.
7. T07 publishes or receives IEC 61850 control traffic.
8. T08 samples runtime evidence.
9. T09 latches TFR metadata on a fault or threshold crossing.
10. T10 flushes recorder payload after the event.
11. T11 and T12 run as non-critical coexistence load.

This chain gives the allocator a real dependency graph instead of a flat list of formulas.

## 7. Code-Ready Implementation Pattern

Until real services exist, use one workload harness process with one thread per task. That is enough to test affinity, priorities, IPC pressure, cache effects, and disturbance handling on i.MX95.

### 7.1 Runtime Primitives

- `pthread_create` for task threads.
- `pthread_setaffinity_np` for static core binding.
- `sched_setscheduler` with `SCHED_FIFO` for RT tasks.
- `clock_nanosleep(..., TIMER_ABSTIME, ...)` for periodic loops.
- `eventfd` for point-to-point event notification.
- cache-aligned shared-memory structs with sequence counters for lock-free snapshots.
- ring buffers for burst paths such as recorder metadata and telemetry export.

### 7.2 Shared Data Blocks

Use data shapes like these instead of a single monolithic state blob.

```c
typedef struct {
    uint64_t seq;
    uint64_t tai_ns;
    float va;
    float vb;
    float vc;
    float ia;
    float ib;
    float ic;
    uint32_t quality;
} measurement_frame_t;

typedef struct {
    uint64_t seq;
    uint64_t tai_ns;
    float vd;
    float vq;
    float id;
    float iq;
    float pll_angle;
    uint32_t protection_flags;
} control_state_t;

typedef struct {
    uint64_t seq;
    uint64_t event_id;
    uint64_t tai_ns;
    uint32_t trigger_flags;
    uint32_t source_task_id;
} tfr_event_t;
```

### 7.3 Thread Descriptor

```c
typedef void (*task_entry_fn)(void *ctx);

typedef struct {
    const char *name;
    int core_id;
    int sched_policy;
    int priority;
    uint32_t period_us;
    task_entry_fn entry;
    void *ctx;
} task_desc_t;
```

### 7.4 Periodic Thread Skeleton

Use absolute-time wakeups. Relative sleeps accumulate drift and hide scheduler problems.

```c
static void run_periodic_task(const task_desc_t *task)
{
    struct timespec next_release;
    clock_gettime(CLOCK_TAI, &next_release);

    for (;;) {
        task->entry(task->ctx);

        next_release.tv_nsec += (long)task->period_us * 1000L;
        while (next_release.tv_nsec >= 1000000000L) {
            next_release.tv_nsec -= 1000000000L;
            next_release.tv_sec += 1;
        }

        clock_nanosleep(CLOCK_TAI, TIMER_ABSTIME, &next_release, NULL);
    }
}
```

### 7.5 Real Task Functions to Implement

The thesis owner should stop feeding only arithmetic kernels and instead implement concrete workload functions with data movement and synchronization.

- `task_measurement_ingest()`:
  - receive or synthesize a frame,
  - stamp it with `CLOCK_TAI`,
  - write to `measurement_frame_t`,
  - notify downstream task.
- `task_signal_conditioning()`:
  - read latest frame,
  - run scaling, median or FIR filters, plausibility checks,
  - publish conditioned state.
- `task_fast_protection()`:
  - evaluate thresholds, stale-data detection, quality loss,
  - trigger `tfr_event_t` when any protection condition trips.
- `task_inner_control()`:
  - execute Clarke/Park or dq control steps,
  - update actuator command structure.
- `task_ipc_broker()`:
  - move state across queues,
  - validate ownership, sequence monotonicity, and queue-depth watermark.
- `task_tfr_trigger()`:
  - capture pre/post-event markers,
  - record metadata immediately,
  - avoid bulk copy in critical context.
- `task_tfr_writer()`:
  - drain event ring,
  - perform buffer compaction or file sink write on non-RT core.
- `task_ai_feature_extract()` and `task_ai_infer()`:
  - consume downsampled state,
  - produce non-critical health output,
  - never block control path.

## 8. What to Measure on i.MX95

Each run should produce evidence that the mapping behaves like a real deployment candidate.

| Metric | Why It Is Needed |
|---|---|
| per-task release jitter | confirms scheduler quality |
| per-task execution time and WCET estimate | confirms core fit |
| deadline miss count | primary acceptance metric |
| queue depth and queue wait | exposes IPC congestion |
| shared-state retry count from sequence checks | exposes writer/reader contention |
| migration count per task | verifies affinity integrity |
| per-core utilization | baseline allocator input |
| cache-miss proxy or context-switch count | useful when two mappings have same mean CPU load |
| TFR trigger latency | proves recorder path does not break the RT chain |
| thermal and power trace | catches mappings that pass only at cold start |

## 9. Disturbance Profiles the Harness Must Run

Do not validate only nominal steady-state load. At minimum, run these profiles:

| Profile | Injected Condition | Expected Stress |
|---|---|---|
| P1 Nominal | steady measurement and control cycles | baseline scheduler behavior |
| P2 Comms burst | burst of GOOSE publish/receive events | IRQ and communication contention |
| P3 Fault event | protection threshold crossing and TFR trigger | recorder start during RT disturbance |
| P4 AI coexistence | inference active during nominal control | best-effort compute interference |
| P5 Combined stress | comms burst plus fault plus telemetry sample window | realistic worst case |
| P6 Thermal soak | long-duration sustained run | late-appearing timing degradation |

## 10. Acceptance Logic for the Thesis

A mapping is realistic only if all of the following remain true during the disturbance profiles above:

- no deadline misses in T01 to T04 and T13 for the accepted operating envelope,
- near-zero migration for critical tasks,
- bounded queue depth for T06 and T07,
- TFR metadata trigger latency remains within the defined fault-recording budget,
- AI and logging workloads never cause priority inversion or starvation in critical tasks,
- measured performance after thermal soak remains inside the same acceptance bounds.
