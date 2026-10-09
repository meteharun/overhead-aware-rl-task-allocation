/* Execution time of the 25 STATCOM tasks, per core and per core class.
 *
 *   ./measure_wcet --iters 100000 > a55_exec_raw.json
 *   ./measure_wcet --cores 0,4 --iters 10000 --task t100_01
 *
 * This is the campaign that produced wcets_isolated.jsonl. It runs
 * each task from statcom_tasks.c back to back with a warm cache and no
 * neighbours, once on every core, and reports the distribution per core and
 * merged per core class.
 *
 * Why per core class. Execution speed is the same on all six cores, so one cost
 * per task is valid anywhere. The tail is not: cores 0-3 take interrupts and
 * cores 4-5 are isolated, so the same code has a different p99.9 depending on
 * where it ran. The timing model needs both, the cost from the median and the
 * tail factor from p99.9 over p50, so both come out of one run here.
 *
 * Core classes follow the board's boot settings, isolcpus=4,5 nohz_full=4,5
 * rcu_nocbs=4,5 with IRQs steered to 0-3:
 *
 *   core 0     os         housekeeping and interrupt handling
 *   cores 1-3  general    ordinary scheduling pool, still takes interrupts
 *   cores 4-5  isolated   nothing else is scheduled there
 *
 * --os, --general and --isolated override that if the board is set up
 * differently.
 *
 * Per class the harness reports the median across that class's cores of each
 * core's p50, and the largest p99.9 any of its cores produced. The median is
 * the cost, the maximum is the tail the model has to survive. Per-core numbers
 * are emitted too, so anything else can be derived without rerunning.
 *
 * What the harness does to make the number mean something:
 *
 *   affinity    pinned to one core at a time
 *   priority    SCHED_FIFO, so nothing but an interrupt preempts the loop
 *   memory      mlockall, so no page fault lands inside a measured window
 *   warm-up     the first --warmup iterations are discarded
 *   clock       CLOCK_MONOTONIC, read either side of the call and nothing else
 *   liveness    the out buffer feeds a volatile sink, so nothing is elided
 *
 * The ARM generic timer runs at 24 MHz, a floor of 41.667 ns. Nothing below one
 * tick is meaningful.
 *
 * Build: make
 * Run as root, or with CAP_SYS_NICE for SCHED_FIFO and CAP_IPC_LOCK for mlock.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define BUFSZ 32
#define MAXCORES 16
#define DEFAULT_ITERS 100000
#define DEFAULT_WARMUP 2000

/* statcom_tasks.c. Declared here so that file stays untouched. */
typedef void (*task_fn)(float *in, float *out);

#define TASKS(X) \
    X(t100_01) X(t100_02) X(t100_03) X(t100_04) X(t100_05) X(t100_06) \
    X(t100_07) X(t100_08) X(t100_09) X(t100_10) X(t100_11) \
    X(t500_01) X(t500_02) X(t500_03) X(t500_04) X(t500_05) X(t500_06) \
    X(t1ms_01) X(t1ms_02) X(t1ms_03) X(t1ms_04) X(t1ms_05) \
    X(t2ms_01) X(t2ms_02) X(t2ms_03)

#define DECLARE(name) void name(float *in, float *out);
TASKS(DECLARE)
#undef DECLARE

static const struct { const char *name; task_fn fn; int period_us; }
TASK[] = {
#define ENTRY(name) {#name, name, 0},
    TASKS(ENTRY)
#undef ENTRY
};
#define NTASK ((int)(sizeof(TASK) / sizeof(TASK[0])))

static int period_of(const char *n)
{
    if (!strncmp(n, "t100_", 5))  return 100;
    if (!strncmp(n, "t500_", 5))  return 500;
    if (!strncmp(n, "t1ms_", 5))  return 1000;
    if (!strncmp(n, "t2ms_", 5))  return 2000;
    return 0;
}

static const char *tier_of(int period)
{
    if (period == 100) return "HARD_RT";
    if (period == 500) return "FIRM_RT";
    return "SOFT";
}

static volatile float sink;

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* linear interpolation between order statistics, in nanoseconds */
static double pct(const uint64_t *sorted, int n, double p)
{
    if (n <= 0) return 0.0;
    double idx = p * (n - 1);
    int lo = (int)idx, hi = lo + 1 < n ? lo + 1 : lo;
    double frac = idx - lo;
    return (1.0 - frac) * sorted[lo] + frac * sorted[hi];
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n)
{
    if (n <= 0) return 0.0;
    qsort(v, (size_t)n, sizeof(double), cmp_dbl);
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static int pin(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "sched_setaffinity(core %d): %s\n", core, strerror(errno));
        return -1;
    }
    return 0;
}

static void go_realtime(void)
{
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = sched_get_priority_max(SCHED_FIFO) - 1;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
        fprintf(stderr, "SCHED_FIFO refused (%s). Run as root or grant "
                        "CAP_SYS_NICE. Numbers will be noisier.\n",
                strerror(errno));
}

/* "0,4" or "1-3" or "0-5" -> a core list. Returns the count, -1 on nonsense. */
static int parse_cores(const char *s, int *out, int max)
{
    int n = 0;
    while (*s && n < max) {
        char *end;
        long a = strtol(s, &end, 10);
        if (end == s) return -1;
        long b = a;
        if (*end == '-') {
            s = end + 1;
            b = strtol(s, &end, 10);
            if (end == s) return -1;
        }
        if (a < 0 || b < a || b >= MAXCORES) return -1;
        for (long c = a; c <= b && n < max; c++) out[n++] = (int)c;
        s = (*end == ',') ? end + 1 : end;
    }
    return n;
}

static int in_list(int core, const int *list, int n)
{
    for (int i = 0; i < n; i++) if (list[i] == core) return 1;
    return 0;
}

static void emit_env(void)
{
    struct utsname u;
    uname(&u);
    char buf[256] = "";
    FILE *f = fopen("/sys/devices/system/cpu/isolated", "r");
    if (f) { if (fgets(buf, sizeof(buf), f)) buf[strcspn(buf, "\n")] = 0; fclose(f); }

    char gov[64] = "unknown";
    f = fopen("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "r");
    if (f) { if (fgets(gov, sizeof(gov), f)) gov[strcspn(gov, "\n")] = 0; fclose(f); }

    char cmdline[512] = "";
    f = fopen("/proc/cmdline", "r");
    if (f) { if (fgets(cmdline, sizeof(cmdline), f)) cmdline[strcspn(cmdline, "\n")] = 0; fclose(f); }

    printf("  \"environment\": {\n");
    printf("    \"kernel\": \"%s %s\",\n", u.sysname, u.release);
    printf("    \"machine\": \"%s\",\n", u.machine);
    printf("    \"governor_cpu0\": \"%s\",\n", gov);
    printf("    \"isolated_cpus\": \"%s\",\n", buf);
    printf("    \"cmdline\": \"%s\"\n", cmdline);
    printf("  },\n");
}

int main(int argc, char **argv)
{
    int iters = DEFAULT_ITERS, warmup = DEFAULT_WARMUP;
    const char *only = NULL;
    int cores[MAXCORES], os[MAXCORES], gen[MAXCORES], iso[MAXCORES];
    int ncore = parse_cores("0-5", cores, MAXCORES);
    int nos = parse_cores("0", os, MAXCORES);
    int ngen = parse_cores("1-3", gen, MAXCORES);
    int niso = parse_cores("4-5", iso, MAXCORES);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(a, "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(a, "--task") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(a, "--cores") && i + 1 < argc)
            ncore = parse_cores(argv[++i], cores, MAXCORES);
        else if (!strcmp(a, "--os") && i + 1 < argc)
            nos = parse_cores(argv[++i], os, MAXCORES);
        else if (!strcmp(a, "--general") && i + 1 < argc)
            ngen = parse_cores(argv[++i], gen, MAXCORES);
        else if (!strcmp(a, "--isolated") && i + 1 < argc)
            niso = parse_cores(argv[++i], iso, MAXCORES);
        else {
            fprintf(stderr,
                    "usage: %s [--iters N] [--warmup N] [--task NAME]\n"
                    "          [--cores 0-5] [--os 0] [--general 1-3] "
                    "[--isolated 4-5]\n", argv[0]);
            return 2;
        }
    }

    if (iters < 1) { fprintf(stderr, "--iters must be at least 1\n"); return 2; }
    if (warmup < 0) { fprintf(stderr, "--warmup cannot be negative\n"); return 2; }
    if (ncore < 1 || nos < 0 || ngen < 0 || niso < 0) {
        fprintf(stderr, "could not read a core list\n");
        return 2;
    }
    if (only) {
        int found = 0;
        for (int t = 0; t < NTASK; t++) if (!strcmp(only, TASK[t].name)) found = 1;
        if (!found) {
            fprintf(stderr, "no task called '%s'. The %d tasks are:\n", only, NTASK);
            for (int t = 0; t < NTASK; t++) fprintf(stderr, "  %s\n", TASK[t].name);
            return 2;
        }
    }

    go_realtime();
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        fprintf(stderr, "mlockall refused (%s). A page fault may land in a "
                        "measured window.\n", strerror(errno));

    uint64_t *s = malloc((size_t)iters * sizeof(uint64_t));
    float *in = malloc(BUFSZ * sizeof(float));
    float *out = malloc(BUFSZ * sizeof(float));
    if (!s || !in || !out) { fprintf(stderr, "out of memory\n"); return 1; }

    printf("{\n");
    emit_env();
    printf("  \"iterations_per_core\": %d,\n", iters);
    printf("  \"warmup\": %d,\n", warmup);
    printf("  \"unit\": \"nanoseconds\",\n");
    printf("  \"method\": \"back-to-back, warm cache, alone on the core, "
           "SCHED_FIFO, memory locked\",\n");
    printf("  \"class_rule\": \"p50 is the median across the class's cores of "
           "each core's p50; p999 is the largest p99.9 any of its cores "
           "produced\",\n");
    printf("  \"measurements\": {\n");

    int printed = 0;
    for (int t = 0; t < NTASK; t++) {
        if (only && strcmp(only, TASK[t].name)) continue;
        int period = period_of(TASK[t].name);

        double core_p50[MAXCORES], core_p999[MAXCORES];
        double cls_p50[3][MAXCORES], cls_p999[3][MAXCORES];
        int ncls[3] = {0, 0, 0};
        double overall_min = 0, overall_max = 0, overall_mean = 0;

        if (printed++) printf(",\n");
        printf("    \"%s\": {\n", TASK[t].name);
        printf("      \"period_us\": %d, \"tier\": \"%s\",\n",
               period, tier_of(period));
        printf("      \"per_core\": {\n");

        for (int c = 0; c < ncore; c++) {
            int core = cores[c];
            if (pin(core) != 0) return 1;

            for (int i = 0; i < BUFSZ; i++) {
                in[i] = 1.0f + 0.01f * i;
                out[i] = 0.0f;
            }
            for (int i = 0; i < warmup; i++) TASK[t].fn(in, out);

            for (int i = 0; i < iters; i++) {
                uint64_t t0 = now_ns();
                TASK[t].fn(in, out);
                uint64_t t1 = now_ns();
                s[i] = t1 - t0;
            }
            sink += out[0];
            qsort(s, (size_t)iters, sizeof(uint64_t), cmp_u64);

            double p50 = pct(s, iters, 0.50), p999 = pct(s, iters, 0.999);
            double sum = 0.0;
            for (int i = 0; i < iters; i++) sum += (double)s[i];
            double mean = sum / iters;

            core_p50[c] = p50;
            core_p999[c] = p999;
            if (c == 0 || s[0] < overall_min) overall_min = s[0];
            if (c == 0 || s[iters - 1] > overall_max) overall_max = s[iters - 1];
            overall_mean += mean / ncore;

            int k = in_list(core, os, nos) ? 0
                  : in_list(core, gen, ngen) ? 1
                  : in_list(core, iso, niso) ? 2 : -1;
            if (k >= 0) {
                cls_p50[k][ncls[k]] = p50;
                cls_p999[k][ncls[k]] = p999;
                ncls[k]++;
            }

            printf("        \"%d\": {\"min\": %.0f, \"p50\": %.1f, \"p90\": %.1f, "
                   "\"p99\": %.1f, \"p999\": %.1f, \"max\": %.0f, \"mean\": %.1f}%s\n",
                   core, (double)s[0], p50, pct(s, iters, 0.90),
                   pct(s, iters, 0.99), p999, (double)s[iters - 1], mean,
                   c + 1 < ncore ? "," : "");
            fprintf(stderr, "%-10s core %d  p50 %9.0f ns  p99.9 %9.0f ns\n",
                    TASK[t].name, core, p50, p999);
        }
        printf("      },\n");

        static const char *CLASS[3] = {"os", "general", "isolated"};
        printf("      \"per_class\": {\n");
        for (int k = 0; k < 3; k++) {
            if (!ncls[k]) continue;
            double p50 = median(cls_p50[k], ncls[k]);
            double worst = cls_p999[k][0];
            for (int i = 1; i < ncls[k]; i++)
                if (cls_p999[k][i] > worst) worst = cls_p999[k][i];
            printf("        \"%s\": {\"cores\": %d, \"p50\": %.1f, "
                   "\"p999\": %.1f, \"tail_ratio\": %.4f}%s\n",
                   CLASS[k], ncls[k], p50, worst, p50 > 0 ? worst / p50 : 0.0,
                   k < 2 && (ncls[k + 1] || ncls[2]) ? "," : "");
        }
        printf("      },\n");

        /* the cost the model uses: speed is core-independent, so one number */
        double all[MAXCORES];
        for (int c = 0; c < ncore; c++) all[c] = core_p50[c];
        double nominal = median(all, ncore);
        double spread = 0.0;
        for (int c = 0; c < ncore; c++) {
            double d = (core_p50[c] - nominal) / (nominal > 0 ? nominal : 1);
            if (d < 0) d = -d;
            if (d > spread) spread = d;
        }
        printf("      \"nominal_ns\": %.1f, \"speed_spread_pct\": %.3f,\n",
               nominal, 100.0 * spread);
        printf("      \"min\": %.0f, \"max\": %.0f, \"mean\": %.1f\n",
               overall_min, overall_max, overall_mean);
        printf("    }");
    }

    printf("\n  }\n}\n");
    free(s); free(in); free(out);
    if (sink == 1234.5f) fprintf(stderr, "");   /* keep the sink live */
    return 0;
}
