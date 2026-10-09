#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <time.h>
#include <unistd.h>

#ifndef CLOCK_TAI
#define CLOCK_TAI CLOCK_MONOTONIC
#endif

#define TASK_COUNT              13
#define PROFILE_SWITCH_NS       2000000000ULL
#define TFR_RING_CAPACITY       64
#define AI_FEATURE_VECTOR_SIZE  64
#define BURST_LOOP_COUNT        96U
#define CONDITIONING_TAPS       5U

typedef enum {
    PROFILE_NOMINAL = 0,
    PROFILE_COMMS,
    PROFILE_FAULT,
    PROFILE_COMBINED,
} workload_profile_t;

typedef struct {
    uint64_t tai_ns;
    float va;
    float vb;
    float vc;
    float ia;
    float ib;
    float ic;
    uint32_t quality;
} measurement_payload_t;

typedef struct {
    _Atomic uint64_t seq;
    measurement_payload_t payload;
} measurement_slot_t;

typedef struct {
    uint64_t tai_ns;
    float vd;
    float vq;
    float id;
    float iq;
    float pll_angle;
    uint32_t protection_flags;
} control_payload_t;

typedef struct {
    _Atomic uint64_t seq;
    control_payload_t payload;
} control_slot_t;

typedef struct {
    uint64_t tai_ns;
    float d_axis_cmd;
    float q_axis_cmd;
    uint32_t command_flags;
} command_payload_t;

typedef struct {
    _Atomic uint64_t seq;
    command_payload_t payload;
} command_slot_t;

typedef struct {
    uint64_t event_id;
    uint64_t tai_ns;
    uint32_t trigger_flags;
    uint32_t source_task_id;
} tfr_event_t;

typedef struct {
    pthread_mutex_t lock;
    tfr_event_t events[TFR_RING_CAPACITY];
    uint32_t head;
    uint32_t tail;
    uint64_t dropped;
} tfr_ring_t;

typedef struct {
    const char* name;
    int core_id;
    int sched_policy;
    int priority;
    uint32_t period_us;
    bool critical;
    uint64_t releases;
    uint64_t deadline_misses;
    uint64_t events;
    int64_t max_exec_ns;
    int64_t max_start_late_ns;
} task_metrics_t;

typedef struct {
    workload_profile_t profile;
    uint64_t duration_ns;
    bool quiet;
    uint32_t cpu_count;
    _Atomic bool running;
    _Atomic uint64_t next_event_id;
    _Atomic bool fault_active;

    measurement_slot_t measurement_slot;
    control_slot_t control_slot;
    command_slot_t command_slot;

    float filter_history[CONDITIONING_TAPS][3];
    uint32_t filter_index;

    tfr_ring_t tfr_ring;
    tfr_event_t pending_event;
    pthread_mutex_t pending_event_lock;
    int tfr_event_fd;

    uint64_t comms_bursts;
    uint64_t telemetry_snapshots;
    uint64_t ai_windows;
} workload_app_t;

typedef struct task_ctx task_ctx_t;
typedef void (*task_fn_t)(task_ctx_t* ctx);

struct task_ctx {
    workload_app_t* app;
    task_metrics_t metrics;
    task_fn_t run;
};

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_TAI, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

static void timespec_add_us(struct timespec* ts, uint32_t period_us)
{
    ts->tv_nsec += (long)period_us * 1000L;
    while (ts->tv_nsec >= 1000000000L) {
        ts->tv_nsec -= 1000000000L;
        ts->tv_sec += 1;
    }
}

static uint64_t timespec_to_ns(const struct timespec* ts)
{
    return ((uint64_t)ts->tv_sec * 1000000000ULL) + (uint64_t)ts->tv_nsec;
}

static uint64_t load_measurement(const measurement_slot_t* slot, measurement_payload_t* out)
{
    uint64_t start_seq;
    uint64_t end_seq;

    do {
        start_seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
        if ((start_seq & 1ULL) != 0ULL) {
            continue;
        }

        *out = slot->payload;
        end_seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
    } while (start_seq != end_seq || (end_seq & 1ULL) != 0ULL);

    return end_seq;
}

static void store_measurement(measurement_slot_t* slot, const measurement_payload_t* payload)
{
    uint64_t seq = atomic_load_explicit(&slot->seq, memory_order_relaxed);
    atomic_store_explicit(&slot->seq, seq + 1ULL, memory_order_release);
    slot->payload = *payload;
    atomic_store_explicit(&slot->seq, seq + 2ULL, memory_order_release);
}

static uint64_t load_control(const control_slot_t* slot, control_payload_t* out)
{
    uint64_t start_seq;
    uint64_t end_seq;

    do {
        start_seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
        if ((start_seq & 1ULL) != 0ULL) {
            continue;
        }

        *out = slot->payload;
        end_seq = atomic_load_explicit(&slot->seq, memory_order_acquire);
    } while (start_seq != end_seq || (end_seq & 1ULL) != 0ULL);

    return end_seq;
}

static void store_control(control_slot_t* slot, const control_payload_t* payload)
{
    uint64_t seq = atomic_load_explicit(&slot->seq, memory_order_relaxed);
    atomic_store_explicit(&slot->seq, seq + 1ULL, memory_order_release);
    slot->payload = *payload;
    atomic_store_explicit(&slot->seq, seq + 2ULL, memory_order_release);
}

static void store_command(command_slot_t* slot, const command_payload_t* payload)
{
    uint64_t seq = atomic_load_explicit(&slot->seq, memory_order_relaxed);
    atomic_store_explicit(&slot->seq, seq + 1ULL, memory_order_release);
    slot->payload = *payload;
    atomic_store_explicit(&slot->seq, seq + 2ULL, memory_order_release);
}

static bool tfr_ring_push(tfr_ring_t* ring, const tfr_event_t* event)
{
    bool ok = false;

    pthread_mutex_lock(&ring->lock);
    uint32_t next_head = (ring->head + 1U) % TFR_RING_CAPACITY;

    if (next_head == ring->tail) {
        ring->dropped++;
    }
    else {
        ring->events[ring->head] = *event;
        ring->head = next_head;
        ok = true;
    }

    pthread_mutex_unlock(&ring->lock);
    return ok;
}

static bool tfr_ring_pop(tfr_ring_t* ring, tfr_event_t* event)
{
    bool ok = false;

    pthread_mutex_lock(&ring->lock);
    if (ring->tail != ring->head) {
        *event = ring->events[ring->tail];
        ring->tail = (ring->tail + 1U) % TFR_RING_CAPACITY;
        ok = true;
    }
    pthread_mutex_unlock(&ring->lock);

    return ok;
}

static void apply_task_affinity_and_sched(task_ctx_t* ctx)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((unsigned)ctx->metrics.core_id, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0 && !ctx->app->quiet) {
        fprintf(stderr, "WARN: could not pin %s to CPU %d\n", ctx->metrics.name, ctx->metrics.core_id);
    }

    struct sched_param sched = { .sched_priority = ctx->metrics.priority };
    if (pthread_setschedparam(pthread_self(), ctx->metrics.sched_policy, &sched) != 0 && !ctx->app->quiet) {
        fprintf(stderr, "WARN: could not set scheduler for %s (need elevated privilege for FIFO)\n", ctx->metrics.name);
    }
}

static void simulate_copy_pressure(uint32_t count)
{
    volatile uint32_t scratch = 0U;
    for (uint32_t i = 0; i < count; ++i) {
        scratch ^= (i * 2654435761U) ^ (scratch << 1U);
    }
    (void)scratch;
}

static void task_measurement_ingest(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t payload;
    uint64_t t_ns = now_ns();
    bool inject_fault = false;

    if (app->profile == PROFILE_FAULT || app->profile == PROFILE_COMBINED) {
        if (ctx->metrics.releases * (uint64_t)ctx->metrics.period_us * 1000ULL >= PROFILE_SWITCH_NS) {
            inject_fault = true;
            atomic_store_explicit(&app->fault_active, true, memory_order_release);
        }
    }

    payload.tai_ns = t_ns;
    payload.va = inject_fault ? 1700.0f : 985.0f + (float)(ctx->metrics.releases % 12U);
    payload.vb = inject_fault ? 1650.0f : 980.0f + (float)(ctx->metrics.releases % 10U);
    payload.vc = inject_fault ? 1680.0f : 978.0f + (float)(ctx->metrics.releases % 8U);
    payload.ia = inject_fault ? 380.0f : 115.0f + (float)(ctx->metrics.releases % 7U);
    payload.ib = inject_fault ? 375.0f : 110.0f + (float)(ctx->metrics.releases % 5U);
    payload.ic = inject_fault ? 385.0f : 113.0f + (float)(ctx->metrics.releases % 6U);
    payload.quality = inject_fault ? 0x0000FFFFU : 0U;

    store_measurement(&app->measurement_slot, &payload);

    if (app->profile == PROFILE_COMMS || app->profile == PROFILE_COMBINED) {
        simulate_copy_pressure(BURST_LOOP_COUNT);
    }

    ctx->metrics.events++;
}

static void task_signal_conditioning(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t payload;
    (void)load_measurement(&app->measurement_slot, &payload);

    app->filter_history[app->filter_index][0] = payload.va;
    app->filter_history[app->filter_index][1] = payload.vb;
    app->filter_history[app->filter_index][2] = payload.vc;
    app->filter_index = (app->filter_index + 1U) % CONDITIONING_TAPS;

    float sum_va = 0.0f;
    float sum_vb = 0.0f;
    float sum_vc = 0.0f;

    for (uint32_t i = 0; i < CONDITIONING_TAPS; ++i) {
        sum_va += app->filter_history[i][0];
        sum_vb += app->filter_history[i][1];
        sum_vc += app->filter_history[i][2];
    }

    payload.va = sum_va / (float)CONDITIONING_TAPS;
    payload.vb = sum_vb / (float)CONDITIONING_TAPS;
    payload.vc = sum_vc / (float)CONDITIONING_TAPS;
    store_measurement(&app->measurement_slot, &payload);
}

static void task_fast_protection(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t payload;
    tfr_event_t event;
    static bool fault_latched = false;

    (void)load_measurement(&app->measurement_slot, &payload);

    bool threshold = (payload.va > 1500.0f) || (payload.ia > 300.0f) || (payload.quality != 0U);
    if (threshold && !fault_latched) {
        fault_latched = true;
        event.event_id = atomic_fetch_add_explicit(&app->next_event_id, 1ULL, memory_order_relaxed) + 1ULL;
        event.tai_ns = now_ns();
        event.trigger_flags = 0x1U;
        event.source_task_id = 3U;

        pthread_mutex_lock(&app->pending_event_lock);
        app->pending_event = event;
        pthread_mutex_unlock(&app->pending_event_lock);

        (void)eventfd_write(app->tfr_event_fd, 1ULL);
        ctx->metrics.events++;
    }
}

static void task_inner_control(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t measurement;
    control_payload_t control;

    (void)load_measurement(&app->measurement_slot, &measurement);

    control.tai_ns = now_ns();
    control.vd = (measurement.va - measurement.vb) * 0.57735f;
    control.vq = (measurement.vb - measurement.vc) * 0.57735f;
    control.id = measurement.ia * 0.85f;
    control.iq = measurement.ib * 0.82f;
    control.pll_angle = (float)((ctx->metrics.releases % 200U) * 0.0314159f);
    control.protection_flags = atomic_load_explicit(&app->fault_active, memory_order_acquire) ? 0x1U : 0U;

    store_control(&app->control_slot, &control);
}

static void task_supervisory(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    if (app->profile == PROFILE_COMMS || app->profile == PROFILE_COMBINED) {
        app->comms_bursts += 1ULL;
        simulate_copy_pressure(BURST_LOOP_COUNT / 2U);
    }
}

static void task_ipc_broker(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t measurement;
    control_payload_t control;

    (void)load_measurement(&app->measurement_slot, &measurement);
    (void)load_control(&app->control_slot, &control);

    if ((measurement.tai_ns > 0ULL) && (control.tai_ns > 0ULL)) {
        simulate_copy_pressure(32U);
        ctx->metrics.events++;
    }
}

static void task_rt_comms(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t measurement;

    (void)load_measurement(&app->measurement_slot, &measurement);
    simulate_copy_pressure((app->profile == PROFILE_COMMS || app->profile == PROFILE_COMBINED) ? BURST_LOOP_COUNT : 24U);
    if (measurement.quality != 0U) {
        ctx->metrics.events++;
    }
}

static void task_telemetry(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t measurement;
    control_payload_t control;

    (void)load_measurement(&app->measurement_slot, &measurement);
    (void)load_control(&app->control_slot, &control);
    app->telemetry_snapshots += 1ULL;
    if ((measurement.tai_ns > 0ULL) && (control.tai_ns > 0ULL)) {
        ctx->metrics.events++;
    }
}

static void task_tfr_writer(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    tfr_event_t event;

    while (tfr_ring_pop(&app->tfr_ring, &event)) {
        simulate_copy_pressure(BURST_LOOP_COUNT * 2U);
        ctx->metrics.events++;
    }
}

static void task_ai_feature(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    measurement_payload_t measurement;
    float features[AI_FEATURE_VECTOR_SIZE];

    (void)load_measurement(&app->measurement_slot, &measurement);
    for (uint32_t i = 0; i < AI_FEATURE_VECTOR_SIZE; ++i) {
        features[i] = (measurement.va + measurement.vb + measurement.vc) * 0.001f + (float)i * 0.01f;
    }
    app->ai_windows += 1ULL;
    ctx->metrics.events += features[0] > 0.0f ? 1ULL : 0ULL;
}

static void task_ai_infer(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    float score = 0.0f;

    for (uint32_t i = 0; i < AI_FEATURE_VECTOR_SIZE; ++i) {
        score += (float)i * 0.015625f;
    }

    if (app->profile == PROFILE_COMBINED) {
        simulate_copy_pressure(BURST_LOOP_COUNT);
    }

    if (score > 0.0f) {
        ctx->metrics.events++;
    }
}

static void task_actuator_commit(task_ctx_t* ctx)
{
    workload_app_t* app = ctx->app;
    control_payload_t control;
    command_payload_t command;

    (void)load_control(&app->control_slot, &control);
    command.tai_ns = now_ns();
    command.d_axis_cmd = control.id * 0.25f;
    command.q_axis_cmd = control.iq * 0.25f;
    command.command_flags = control.protection_flags;
    store_command(&app->command_slot, &command);
}

static void* periodic_thread_main(void* arg)
{
    task_ctx_t* ctx = (task_ctx_t*)arg;
    workload_app_t* app = ctx->app;
    struct timespec next_release;

    apply_task_affinity_and_sched(ctx);
    clock_gettime(CLOCK_TAI, &next_release);

    while (atomic_load_explicit(&app->running, memory_order_acquire)) {
        uint64_t release_ns = timespec_to_ns(&next_release);
        (void)clock_nanosleep(CLOCK_TAI, TIMER_ABSTIME, &next_release, NULL);

        uint64_t start_ns = now_ns();
        int64_t start_late_ns = (int64_t)(start_ns - release_ns);
        if (start_late_ns > ctx->metrics.max_start_late_ns) {
            ctx->metrics.max_start_late_ns = start_late_ns;
        }

        ctx->run(ctx);
        ctx->metrics.releases++;

        uint64_t finish_ns = now_ns();
        int64_t exec_ns = (int64_t)(finish_ns - start_ns);
        if (exec_ns > ctx->metrics.max_exec_ns) {
            ctx->metrics.max_exec_ns = exec_ns;
        }

        if (finish_ns > (release_ns + ((uint64_t)ctx->metrics.period_us * 1000ULL))) {
            ctx->metrics.deadline_misses++;
        }

        timespec_add_us(&next_release, ctx->metrics.period_us);
    }

    return NULL;
}

static void* tfr_trigger_thread_main(void* arg)
{
    task_ctx_t* ctx = (task_ctx_t*)arg;
    workload_app_t* app = ctx->app;

    apply_task_affinity_and_sched(ctx);

    while (atomic_load_explicit(&app->running, memory_order_acquire)) {
        eventfd_t count = 0;
        int rc = eventfd_read(app->tfr_event_fd, &count);
        if (rc != 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        if (!atomic_load_explicit(&app->running, memory_order_acquire)) {
            break;
        }

        uint64_t start_ns = now_ns();
        pthread_mutex_lock(&app->pending_event_lock);
        tfr_event_t event = app->pending_event;
        pthread_mutex_unlock(&app->pending_event_lock);

        if (tfr_ring_push(&app->tfr_ring, &event)) {
            ctx->metrics.events += (uint64_t)count;
        }
        ctx->metrics.releases += (uint64_t)count;

        int64_t exec_ns = (int64_t)(now_ns() - start_ns);
        if (exec_ns > ctx->metrics.max_exec_ns) {
            ctx->metrics.max_exec_ns = exec_ns;
        }
    }

    return NULL;
}

static workload_profile_t parse_profile(const char* text)
{
    if (strcmp(text, "nominal") == 0) {
        return PROFILE_NOMINAL;
    }
    if (strcmp(text, "comms") == 0) {
        return PROFILE_COMMS;
    }
    if (strcmp(text, "fault") == 0) {
        return PROFILE_FAULT;
    }
    if (strcmp(text, "combined") == 0) {
        return PROFILE_COMBINED;
    }

    fprintf(stderr, "Unknown profile '%s'\n", text);
    exit(2);
}

static void init_task(task_ctx_t* task,
                      workload_app_t* app,
                      const char* name,
                      int core_id,
                      int policy,
                      int priority,
                      uint32_t period_us,
                      bool critical,
                      task_fn_t run)
{
    memset(task, 0, sizeof(*task));
    task->app = app;
    task->run = run;
    task->metrics.name = name;
    task->metrics.core_id = core_id;
    task->metrics.sched_policy = policy;
    task->metrics.priority = priority;
    task->metrics.period_us = period_us;
    task->metrics.critical = critical;
}

static void print_usage(const char* program)
{
    fprintf(stderr, "Usage: %s [--duration SEC] [--profile nominal|comms|fault|combined] [--quiet]\n", program);
}

int main(int argc, char* argv[])
{
    workload_app_t app;
    pthread_t threads[TASK_COUNT];
    task_ctx_t tasks[TASK_COUNT];
    uint32_t created_threads = 0U;
    int exit_code = 0;

    memset(&app, 0, sizeof(app));
    app.profile = PROFILE_COMBINED;
    app.duration_ns = 5000000000ULL;
    app.cpu_count = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    if (app.cpu_count == 0U) {
        app.cpu_count = 1U;
    }

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--duration") == 0 && (i + 1) < argc) {
            app.duration_ns = (uint64_t)strtoull(argv[++i], NULL, 10) * 1000000000ULL;
        }
        else if (strcmp(argv[i], "--profile") == 0 && (i + 1) < argc) {
            app.profile = parse_profile(argv[++i]);
        }
        else if (strcmp(argv[i], "--quiet") == 0) {
            app.quiet = true;
        }
        else {
            print_usage(argv[0]);
            return 2;
        }
    }

    pthread_mutex_init(&app.pending_event_lock, NULL);
    pthread_mutex_init(&app.tfr_ring.lock, NULL);
    app.tfr_event_fd = eventfd(0, 0);
    if (app.tfr_event_fd < 0) {
        perror("eventfd");
        return 1;
    }

    atomic_store_explicit(&app.running, true, memory_order_release);

    init_task(&tasks[0], &app, "T01_measurement_ingest", 2 % (int)app.cpu_count, SCHED_FIFO, 90, 100U, true, task_measurement_ingest);
    init_task(&tasks[1], &app, "T02_signal_conditioning", 2 % (int)app.cpu_count, SCHED_FIFO, 88, 100U, true, task_signal_conditioning);
    init_task(&tasks[2], &app, "T03_fast_protection", 2 % (int)app.cpu_count, SCHED_FIFO, 92, 100U, true, task_fast_protection);
    init_task(&tasks[3], &app, "T04_inner_control", 3 % (int)app.cpu_count, SCHED_FIFO, 91, 1000U, true, task_inner_control);
    init_task(&tasks[4], &app, "T05_supervisory", 1 % (int)app.cpu_count, SCHED_FIFO, 70, 5000U, false, task_supervisory);
    init_task(&tasks[5], &app, "T06_ipc_broker", 4 % (int)app.cpu_count, SCHED_FIFO, 75, 1000U, false, task_ipc_broker);
    init_task(&tasks[6], &app, "T07_rt_comms", 4 % (int)app.cpu_count, SCHED_FIFO, 80, 1000U, false, task_rt_comms);
    init_task(&tasks[7], &app, "T08_telemetry", 5 % (int)app.cpu_count, SCHED_OTHER, 0, 10000U, false, task_telemetry);
    init_task(&tasks[8], &app, "T09_tfr_trigger", 4 % (int)app.cpu_count, SCHED_FIFO, 76, 0U, false, NULL);
    init_task(&tasks[9], &app, "T10_tfr_writer", 5 % (int)app.cpu_count, SCHED_OTHER, 0, 10000U, false, task_tfr_writer);
    init_task(&tasks[10], &app, "T11_ai_feature", 5 % (int)app.cpu_count, SCHED_OTHER, 0, 20000U, false, task_ai_feature);
    init_task(&tasks[11], &app, "T12_ai_infer", 5 % (int)app.cpu_count, SCHED_OTHER, 0, 100000U, false, task_ai_infer);
    init_task(&tasks[12], &app, "T13_actuator_commit", 3 % (int)app.cpu_count, SCHED_FIFO, 89, 1000U, true, task_actuator_commit);

    printf("STATCOM i.MX95 workload runner\n");
    printf("Profile    : %s\n",
           app.profile == PROFILE_NOMINAL ? "nominal" :
           app.profile == PROFILE_COMMS ? "comms" :
           app.profile == PROFILE_FAULT ? "fault" : "combined");
    printf("Duration   : %.2f s\n", (double)app.duration_ns / 1000000000.0);
    printf("CPU count  : %u\n", app.cpu_count);
    printf("Note       : SCHED_FIFO pinning may require sudo/root on target Linux\n\n");

    for (uint32_t i = 0; i < TASK_COUNT; ++i) {
        int rc;
        if (i == 8U) {
            rc = pthread_create(&threads[i], NULL, tfr_trigger_thread_main, &tasks[i]);
        }
        else {
            rc = pthread_create(&threads[i], NULL, periodic_thread_main, &tasks[i]);
        }

        if (rc != 0) {
            fprintf(stderr, "Failed to create %s: %s\n", tasks[i].metrics.name, strerror(rc));
            exit_code = 1;
            goto cleanup;
        }
        created_threads++;
    }

    struct timespec runtime;
    runtime.tv_sec = (time_t)(app.duration_ns / 1000000000ULL);
    runtime.tv_nsec = (long)(app.duration_ns % 1000000000ULL);
    (void)nanosleep(&runtime, NULL);

cleanup:
    atomic_store_explicit(&app.running, false, memory_order_release);
    (void)eventfd_write(app.tfr_event_fd, 1ULL);

    for (uint32_t i = 0; i < created_threads; ++i) {
        (void)pthread_join(threads[i], NULL);
    }

    printf("Task summary\n");
    printf("---------------------------------------------------------------------------------------------------------------\n");
    printf("%-24s %-4s %-8s %-10s %-10s %-12s %-12s\n",
           "Task", "CPU", "Period", "Releases", "Misses", "MaxExec(us)", "MaxLate(us)");
    printf("---------------------------------------------------------------------------------------------------------------\n");

    for (uint32_t i = 0; i < TASK_COUNT; ++i) {
        printf("%-24s %-4d %-8u %-10" PRIu64 " %-10" PRIu64 " %-12.2f %-12.2f\n",
               tasks[i].metrics.name,
               tasks[i].metrics.core_id,
               tasks[i].metrics.period_us,
               tasks[i].metrics.releases,
               tasks[i].metrics.deadline_misses,
               (double)tasks[i].metrics.max_exec_ns / 1000.0,
               (double)tasks[i].metrics.max_start_late_ns / 1000.0);

        if (tasks[i].metrics.critical && tasks[i].metrics.deadline_misses > 0U) {
            exit_code = 3;
        }
    }

    printf("---------------------------------------------------------------------------------------------------------------\n");
    printf("TFR dropped events   : %" PRIu64 "\n", app.tfr_ring.dropped);
    printf("Comms bursts         : %" PRIu64 "\n", app.comms_bursts);
    printf("Telemetry snapshots  : %" PRIu64 "\n", app.telemetry_snapshots);
    printf("AI windows           : %" PRIu64 "\n", app.ai_windows);

    close(app.tfr_event_fd);
    pthread_mutex_destroy(&app.pending_event_lock);
    pthread_mutex_destroy(&app.tfr_ring.lock);

    return exit_code;
}