/* ============================================================================
 * measure_alloc_own.c -- run real allocations and measure what each task OWNS.
 *
 * WHY THIS EXISTS
 *
 * measure_tasks.c `alloc` mode records two things per job:
 *     lat   = release -> start          (includes waiting for higher-priority work)
 *     exec  = start -> end, WALL CLOCK  (includes any preempter that cut in)
 *
 * Neither is a property of the task. This tool measures the two that are:
 *
 *     exec_own = the CPU time THIS task consumed, preempters excluded
 *     lat_own  = the gap from when this task COULD start to when it did,
 *                blocking excluded
 *
 * so the model in docs/timing_model_formula.md can be checked against the
 * board rather than inferred from wall-clock arithmetic.
 *
 * HOW exec_own IS MEASURED
 *
 * clock_gettime(CLOCK_THREAD_CPUTIME_ID) returns the CPU time consumed by the
 * calling thread and nothing else. Read it either side of the task body and
 * the preempters are gone -- the kernel does the subtraction, we do not. This
 * is the whole reason the tool can exist.
 *
 *     cpu1 - cpu0  = exec_own        (this thread on the CPU)
 *     t1   - t0    = exec, wall      (what measure_tasks.c reported)
 *     the difference = every preempter's own execution + any idle in between
 *
 * HOW lat_own IS MEASURED
 *
 * A thread cannot see when the core became free, so the threads tell each
 * other: every job publishes its completion time into a per-core slot, and a
 * starting job reads it.
 *
 *     freed  = core_last_end[core]        (most recent completion on this core)
 *     could  = max(release, freed)        (when the CPU was mine to take)
 *     lat_own = start - could
 *
 * and the same comparison says WHICH constant this job measured:
 *
 *     freed <= release  ->  nothing was running when I was released; the timer
 *                           woke the core            -> this job measures D_wake
 *     freed >  release  ->  I was already runnable and waiting for the CPU
 *                           -> this job measures D_disp
 *
 * The two are accumulated into SEPARATE distributions, so one run gives both
 * constants directly instead of one blended number.
 *
 * THE RELEASE GRID IS SHARED -- THIS IS A FIX
 *
 * In measure_tasks.c every thread calls clock_gettime AFTER the barrier and
 * builds its own release grid from it. Threads leave a barrier microseconds
 * apart, so each task's `lat` is measured against a different origin, and any
 * arithmetic across two tasks is wrong by that offset. On the existing dataset
 * lat_i - lat_j - exec_j comes out at -5.09 us over 566 rows -- a task
 * apparently starting before the task above it finished.
 *
 * Here ONE origin is stamped once and every thread uses it, so cross-task
 * arithmetic is valid and lat_own is meaningful.
 *
 * It is stamped AFTER warm-up, not before the threads start. Each thread warms
 * up with 2000 calls, which for a 1 ms task is 2 seconds; an origin stamped
 * before that is already long past when the threads reach it, the first release
 * is a second late, and the catch-up guard skips ~2600 periods before the run
 * settles. Two barriers: the first means "everyone has warmed up", one thread
 * then stamps the origin, the second releases everyone onto it.
 *
 * Build (on the board):
 *   gcc -O2 -std=gnu11 -pthread -o measure_alloc_own \
 *       measure_alloc_own.c statcom_tasks.c -lm -lrt
 *
 * -O2 always, same as every other tool here.
 *
 * Usage:
 *   ./measure_alloc_own <secs_per_alloc> <spec_file>
 *
 * Spec format is measure_tasks.c's, unchanged:
 *   ALLOC <set_index> <method> <n_tasks>
 *   <task_name> <core> <priority> <period_us> <deadline_us>
 *   ... n_tasks lines ...
 *
 * One JSON line per task per allocation on stdout.
 * ==========================================================================*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdatomic.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/resource.h>

#define BUFSZ           32
#define N_TASKS         25
#define MAX_CORES       16
#define ALLOC_MAX_TASKS 64
#define MAX_SAMPLES     2000000

/* Priorities sit BELOW the PREEMPT_RT interrupt threads at FIFO 50. A task set
   that outranks interrupt handling has taken this board down twice. */
#define ALLOC_PRIO_BASE 10

/* ------------------------------------------------------- task catalogue */

#define TASK(n) void n(float*, float*);
TASK(t100_01) TASK(t100_02) TASK(t100_03) TASK(t100_04) TASK(t100_05)
TASK(t100_06) TASK(t100_07) TASK(t100_08) TASK(t100_09) TASK(t100_10)
TASK(t100_11)
TASK(t500_01) TASK(t500_02) TASK(t500_03) TASK(t500_04) TASK(t500_05)
TASK(t500_06)
TASK(t1ms_01) TASK(t1ms_02) TASK(t1ms_03) TASK(t1ms_04) TASK(t1ms_05)
TASK(t2ms_01) TASK(t2ms_02) TASK(t2ms_03)
#undef TASK

typedef void (*task_fn)(float *, float *);

static task_fn TASK_FNS[N_TASKS] = {
    t100_01, t100_02, t100_03, t100_04, t100_05,
    t100_06, t100_07, t100_08, t100_09, t100_10, t100_11,
    t500_01, t500_02, t500_03, t500_04, t500_05, t500_06,
    t1ms_01, t1ms_02, t1ms_03, t1ms_04, t1ms_05,
    t2ms_01, t2ms_02, t2ms_03
};
static const char *TASK_NAMES[N_TASKS] = {
    "t100_01","t100_02","t100_03","t100_04","t100_05",
    "t100_06","t100_07","t100_08","t100_09","t100_10","t100_11",
    "t500_01","t500_02","t500_03","t500_04","t500_05","t500_06",
    "t1ms_01","t1ms_02","t1ms_03","t1ms_04","t1ms_05",
    "t2ms_01","t2ms_02","t2ms_03"
};
static const char *TASK_TIER[N_TASKS] = {
    "HARD_RT","HARD_RT","HARD_RT","HARD_RT","HARD_RT","HARD_RT",
    "HARD_RT","HARD_RT","HARD_RT","HARD_RT","HARD_RT",
    "FIRM_RT","FIRM_RT","FIRM_RT","FIRM_RT","FIRM_RT","FIRM_RT",
    "SOFT","SOFT","SOFT","SOFT","SOFT",
    "SOFT","SOFT","SOFT"
};

/* ---------------------------------------------------------------- utils */

static volatile uint64_t g_sink = 0;

static inline uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* CPU time of the CALLING THREAD only. This is what makes exec_own possible:
   time the thread was not scheduled -- because a preempter had the core -- is
   not counted by the kernel, so we never have to subtract it ourselves. */
static inline uint64_t thread_cpu_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static int pin_to(int core) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s);
    return sched_setaffinity(0, sizeof(s), &s);
}

static long nivcsw(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_THREAD, &ru) != 0) return -1;
    return ru.ru_nivcsw;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

typedef struct {
    long     n;
    uint64_t min, p50, p90, p99, p999, max;
    double   mean, sd;
} stats_t;

static stats_t summarize(uint64_t *v, long n) {
    stats_t s; memset(&s, 0, sizeof(s));
    if (n <= 0) return s;
    qsort(v, (size_t)n, sizeof(uint64_t), cmp_u64);
    double sum = 0.0;
    for (long i = 0; i < n; i++) sum += (double)v[i];
    s.mean = sum / (double)n;
    double var = 0.0;
    for (long i = 0; i < n; i++) { double d = (double)v[i] - s.mean; var += d * d; }
    s.sd  = (n > 1) ? sqrt(var / (double)(n - 1)) : 0.0;
    s.n   = n;
    s.min = v[0];
    s.max = v[n - 1];
    s.p50  = v[(long)(0.50  * (double)(n - 1))];
    s.p90  = v[(long)(0.90  * (double)(n - 1))];
    s.p99  = v[(long)(0.99  * (double)(n - 1))];
    s.p999 = v[(long)(0.999 * (double)(n - 1))];
    return s;
}

static void emit(const char *pfx, stats_t s) {
    printf(",\"%s_n\":%ld,\"%s_min_ns\":%llu,\"%s_mean_ns\":%.1f"
           ",\"%s_sd_ns\":%.1f,\"%s_p50_ns\":%llu,\"%s_p90_ns\":%llu"
           ",\"%s_p99_ns\":%llu,\"%s_p999_ns\":%llu,\"%s_max_ns\":%llu",
           pfx, s.n, pfx, (unsigned long long)s.min, pfx, s.mean,
           pfx, s.sd, pfx, (unsigned long long)s.p50,
           pfx, (unsigned long long)s.p90, pfx, (unsigned long long)s.p99,
           pfx, (unsigned long long)s.p999, pfx, (unsigned long long)s.max);
}

/* ------------------------------------------------ per-core shared state */

/* The most recent job COMPLETION on each core. A starting job reads this to
   learn whether the core was busy when it was released. Relaxed ordering is
   enough: this is a measurement hint, not a synchronisation primitive, and a
   stale read costs one sample's classification, not correctness. */
static _Atomic uint64_t g_core_last_end[MAX_CORES];

/* ---------------------------------------------------------- the threads */

typedef struct {
    int      idx;
    int      core;
    int      prio;
    long     period_us;
    double   deadline_us;
    char     name[40];

    long     n;
    long     overruns, skipped;
    long     n_woke, n_disp;      /* how each job's lat_own was classified */

    /* what the task owns */
    uint64_t *v_lat_own_wake;     /* lat_own, jobs the timer woke  -> D_wake  */
    uint64_t *v_lat_own_disp;     /* lat_own, jobs handed the CPU  -> D_disp  */
    uint64_t *v_exec_own;         /* thread CPU time                          */
    /* what measure_tasks.c would have reported, for continuity */
    uint64_t *v_lat, *v_exec, *v_rt, *v_pre;
} atask_t;

/* TWO barriers, and the reason matters.
 *
 * The origin cannot be stamped before the threads are created: each one warms
 * up with 2000 calls first, which for a 1 ms task is 2 SECONDS. Stamp it early
 * and the first release is already a second in the past, the catch-up guard
 * fires, and ~2600 periods are skipped before the run settles.
 *
 * So: barrier 1 means "everyone has finished warming up". The one thread that
 * pthread_barrier_wait picks as serial then stamps the origin, and barrier 2
 * releases everybody onto it. Shared grid, and no staleness. */
static pthread_barrier_t g_barrier;    /* warm-up complete   */
static pthread_barrier_t g_barrier2;   /* origin is stamped  */
static long              g_secs   = 5;
static _Atomic uint64_t  g_origin = 0; /* SHARED release origin */

static void *task_thread(void *arg) {
    atask_t *t = (atask_t *)arg;
    float in[BUFSZ], out[BUFSZ];
    for (int i = 0; i < BUFSZ; i++) { in[i] = 1.0f + 0.1f * i; out[i] = 0.0f; }
    task_fn fn = TASK_FNS[t->idx];

    pin_to(t->core);
    { struct sched_param sp = { .sched_priority = t->prio };
      sched_setscheduler(0, SCHED_FIFO, &sp); }
    mlockall(MCL_CURRENT | MCL_FUTURE);
    for (long i = 0; i < 2000; i++) fn(in, out);      /* warm */

    /* Phase 1: everyone has warmed up. */
    int rc = pthread_barrier_wait(&g_barrier);
    /* Exactly one thread gets PTHREAD_BARRIER_SERIAL_THREAD. It stamps the
       origin NOW, with every warm-up already paid for. 2 ms is ample for the
       others to get from barrier 2 to their first clock_nanosleep. */
    if (rc == PTHREAD_BARRIER_SERIAL_THREAD)
        atomic_store(&g_origin, now_ns() + 2000000ull);
    /* Phase 2: release everyone onto that origin. */
    pthread_barrier_wait(&g_barrier2);

    uint64_t period_ns = (uint64_t)t->period_us * 1000ull;
    /* Every thread counts from the SAME origin, so releases really are
       simultaneous and cross-task arithmetic is valid. */
    uint64_t target   = atomic_load(&g_origin) + period_ns;
    uint64_t deadline = now_ns() + (uint64_t)g_secs * 1000000000ull;

    long iters = (g_secs * 1000000L) / t->period_us;
    if (iters > MAX_SAMPLES) iters = MAX_SAMPLES;

    for (long i = 0; i < iters; i++) {
        struct timespec next;
        next.tv_sec  = (time_t)(target / 1000000000ull);
        next.tv_nsec = (long)  (target % 1000000000ull);
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);

        /* Who had the core before me? */
        uint64_t freed = atomic_load_explicit(&g_core_last_end[t->core],
                                              memory_order_relaxed);
        long     sw0   = nivcsw();
        uint64_t start = now_ns();
        uint64_t cpu0  = thread_cpu_ns();

        in[0] = 1.0f + (float)(i & 0x3ff) * 0.001f;
        fn(in, out);

        uint64_t cpu1  = thread_cpu_ns();
        uint64_t end   = now_ns();
        long     sw1   = nivcsw();

        atomic_store_explicit(&g_core_last_end[t->core], end,
                              memory_order_relaxed);

        /* ---- the two quantities that belong to this task ---------------- */
        uint64_t could   = (freed > target) ? freed : target;
        uint64_t lat_own = (start > could) ? (start - could) : 0;

        if (freed <= target) {
            /* Core was not busy at my release: the timer woke it. D_wake. */
            t->v_lat_own_wake[t->n_woke++] = lat_own;
        } else {
            /* I was runnable and waiting; the CPU was handed to me. D_disp. */
            t->v_lat_own_disp[t->n_disp++] = lat_own;
        }
        t->v_exec_own[t->n] = cpu1 - cpu0;

        /* ---- what the old tool would have reported ---------------------- */
        t->v_lat[t->n]  = (start > target) ? (start - target) : 0;
        t->v_exec[t->n] = end - start;
        t->v_rt[t->n]   = (end > target) ? (end - target) : 0;
        t->v_pre[t->n]  = (sw1 >= sw0) ? (uint64_t)(sw1 - sw0) : 0;
        t->n++;

        if (end > target + period_ns) t->overruns++;

        /* Catch-up guard: without it an overrunning job turns this loop into
           an unthrottled SCHED_FIFO spin. Skip to the next FUTURE release and
           count what was lost. */
        target += period_ns;
        uint64_t nn = now_ns();
        if (target <= nn) {
            uint64_t miss = (nn - target) / period_ns + 1;
            target     += miss * period_ns;
            t->skipped += (long)miss;
        }
        if ((i & 0x3F) == 0 && nn > deadline) break;
    }
    g_sink += (uint64_t)out[0];
    return NULL;
}

/* --------------------------------------------------------- one allocation */

static void run_alloc(atask_t *ts, int ntasks, int set_index,
                      const char *method) {
    pthread_t th[ALLOC_MAX_TASKS];

    for (int i = 0; i < ntasks; i++) {
        long it = (g_secs * 1000000L) / ts[i].period_us;
        if (it > MAX_SAMPLES) it = MAX_SAMPLES;
        if (it < 1) it = 1;
        ts[i].n = ts[i].overruns = ts[i].skipped = 0;
        ts[i].n_woke = ts[i].n_disp = 0;
        ts[i].v_lat_own_wake = malloc((size_t)it * sizeof(uint64_t));
        ts[i].v_lat_own_disp = malloc((size_t)it * sizeof(uint64_t));
        ts[i].v_exec_own     = malloc((size_t)it * sizeof(uint64_t));
        ts[i].v_lat          = malloc((size_t)it * sizeof(uint64_t));
        ts[i].v_exec         = malloc((size_t)it * sizeof(uint64_t));
        ts[i].v_rt           = malloc((size_t)it * sizeof(uint64_t));
        ts[i].v_pre          = malloc((size_t)it * sizeof(uint64_t));
        if (!ts[i].v_lat_own_wake || !ts[i].v_lat_own_disp ||
            !ts[i].v_exec_own || !ts[i].v_lat || !ts[i].v_exec ||
            !ts[i].v_rt || !ts[i].v_pre) {
            fprintf(stderr, "alloc %d: out of memory for %s\n",
                    set_index, ts[i].name);
            return;
        }
    }

    for (int c = 0; c < MAX_CORES; c++)
        atomic_store(&g_core_last_end[c], 0);

    /* The origin is stamped inside the threads, after warm-up -- see the note
       on g_barrier. Nothing to set here. */
    atomic_store(&g_origin, 0);

    pthread_barrier_init(&g_barrier,  NULL, (unsigned)ntasks);
    pthread_barrier_init(&g_barrier2, NULL, (unsigned)ntasks);
    for (int i = 0; i < ntasks; i++)
        pthread_create(&th[i], NULL, task_thread, &ts[i]);
    for (int i = 0; i < ntasks; i++) pthread_join(th[i], NULL);
    pthread_barrier_destroy(&g_barrier);
    pthread_barrier_destroy(&g_barrier2);

    for (int i = 0; i < ntasks; i++) {
        atask_t *t = &ts[i];
        if (t->n < 1) t->n = 1;

        stats_t s_lw = summarize(t->v_lat_own_wake, t->n_woke);
        stats_t s_ld = summarize(t->v_lat_own_disp, t->n_disp);
        stats_t s_eo = summarize(t->v_exec_own, t->n);
        stats_t s_l  = summarize(t->v_lat,  t->n);
        stats_t s_e  = summarize(t->v_exec, t->n);
        stats_t s_r  = summarize(t->v_rt,   t->n);
        stats_t s_p  = summarize(t->v_pre,  t->n);

        long missed = 0;
        for (long k = 0; k < t->n; k++)
            if ((double)t->v_rt[k] > t->deadline_us * 1000.0) missed++;

        printf("{\"mode\":\"alloc_own\",\"set_index\":%d,\"method\":\"%s\""
               ",\"task\":\"%s\",\"catalogue\":\"%s\",\"core\":%d,\"prio\":%d"
               ",\"tier\":\"%s\",\"period_us\":%ld,\"deadline_us\":%.2f"
               ",\"n_tasks\":%d,\"runs\":%ld,\"overruns\":%ld,\"skipped\":%ld"
               ",\"deadline_misses\":%ld,\"miss_rate_pct\":%.4f"
               ",\"n_woke\":%ld,\"n_disp\":%ld",
               set_index, method, t->name, TASK_NAMES[t->idx], t->core,
               t->prio, TASK_TIER[t->idx], t->period_us, t->deadline_us,
               ntasks, t->n, t->overruns, t->skipped, missed,
               100.0 * (double)missed / (double)t->n,
               t->n_woke, t->n_disp);

        /* what the task owns -- the point of this tool */
        emit("lat_own_wake", s_lw);   /* -> D_wake  */
        emit("lat_own_disp", s_ld);   /* -> D_disp  */
        emit("exec_own",     s_eo);   /* -> C + O_wake + K x R_resume */
        /* what measure_tasks.c reported, so the two datasets are comparable */
        emit("lat",     s_l);
        emit("exec",    s_e);
        emit("rt",      s_r);
        emit("preempt", s_p);
        printf("}\n");

        free(t->v_lat_own_wake); free(t->v_lat_own_disp); free(t->v_exec_own);
        free(t->v_lat); free(t->v_exec); free(t->v_rt); free(t->v_pre);
    }
    fflush(stdout);
}

/* ---------------------------------------------------------------- main */

static int task_index(const char *name) {
    for (int i = 0; i < N_TASKS; i++)
        if (!strcmp(TASK_NAMES[i], name)) return i;
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <secs_per_alloc> <spec_file> [skip_n]\n\n"
            "  Runs each allocation in the spec file and reports, per task:\n"
            "    lat_own_wake_*  lat_own for jobs the timer woke   -> D_wake\n"
            "    lat_own_disp_*  lat_own for jobs handed the CPU   -> D_disp\n"
            "    exec_own_*      CPU time of this thread alone     -> C + O_wake + K.R_resume\n"
            "    lat_* exec_* rt_* preempt_*   as measure_tasks.c reports them\n\n"
            "  skip_n  skip the first N allocations in the spec and run the\n"
            "          rest. Exactly one JSON line is emitted per allocation\n"
            "          and stdout is flushed after each, so a run interrupted\n"
            "          at any point resumes with skip_n = lines already\n"
            "          written. A 10000-allocation sweep is ~17 hours; losing\n"
            "          it to one reboot is not acceptable.\n",
            argv[0]);
        return 2;
    }
    g_secs = atol(argv[1]);
    const char *path = argv[2];
    long skip = (argc > 3) ? atol(argv[3]) : 0;
    long seen = 0;
    if (skip > 0)
        fprintf(stderr, "[alloc_own] resuming: skipping the first %ld "
                        "allocation(s)\n", skip);

    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
              return 2; }

    /* The main thread only spawns and joins; keep it below every task. */
    { struct sched_param sp = { .sched_priority = ALLOC_PRIO_BASE - 1 };
      sched_setscheduler(0, SCHED_FIFO, &sp); }

    char line[256];
    int done = 0;
    while (fgets(line, sizeof(line), f)) {
        int set_index, ntasks;
        char method[32];
        if (sscanf(line, "ALLOC %d %31s %d", &set_index, method, &ntasks) != 3)
            continue;
        if (ntasks < 1 || ntasks > ALLOC_MAX_TASKS) {
            fprintf(stderr, "alloc %d: bad task count %d\n", set_index, ntasks);
            return 2;
        }
        /* Already measured on an earlier attempt: consume its task lines and
           move on, so the spec stays the single source of ordering. */
        if (++seen <= skip) {
            for (int i = 0; i < ntasks; i++)
                if (!fgets(line, sizeof(line), f)) break;
            continue;
        }
        atask_t ts[ALLOC_MAX_TASKS];
        memset(ts, 0, sizeof(ts));
        int ok = 1, maxp = 0;
        for (int i = 0; i < ntasks; i++) {
            char nm[40]; int core, prio; long per; double dl;
            if (!fgets(line, sizeof(line), f) ||
                sscanf(line, "%39s %d %d %ld %lf", nm, &core, &prio, &per, &dl) != 5) {
                fprintf(stderr, "alloc %d: bad task line %d\n", set_index, i);
                ok = 0; break;
            }
            char base[40];
            snprintf(base, sizeof(base), "%s", nm);
            char *hash = strchr(base, '#');
            if (hash) *hash = '\0';
            int idx = task_index(base);
            if (idx < 0) {
                fprintf(stderr, "alloc %d: unknown task %s\n", set_index, base);
                ok = 0; break;
            }
            if (core < 0 || core >= MAX_CORES) {
                fprintf(stderr, "alloc %d: core %d out of range\n", set_index, core);
                ok = 0; break;
            }
            ts[i].idx = idx; ts[i].core = core; ts[i].period_us = per;
            ts[i].deadline_us = dl; ts[i].prio = prio;
            snprintf(ts[i].name, sizeof(ts[i].name), "%s", nm);
            if (prio > maxp) maxp = prio;
        }
        if (!ok) return 2;

        for (int i = 0; i < ntasks; i++)
            ts[i].prio = ALLOC_PRIO_BASE + ts[i].prio;
        if (ALLOC_PRIO_BASE + maxp >= 50)
            fprintf(stderr, "alloc %d: WARNING top priority %d reaches the IRQ "
                            "thread band (50)\n", set_index,
                            ALLOC_PRIO_BASE + maxp);

        run_alloc(ts, ntasks, set_index, method);
        done++;
        if ((done % 25) == 0)
            fprintf(stderr, "[alloc_own] %d done this session, "
                            "%ld of the spec covered\n", done, seen);
    }
    fclose(f);
    fprintf(stderr, "[alloc_own] ran %d allocation(s) this session, "
                    "%ld of the spec covered\n", done, seen);
    return 0;
}
