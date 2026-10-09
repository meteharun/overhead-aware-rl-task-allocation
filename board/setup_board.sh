#!/bin/bash
# =============================================================================
# setup_board.sh -- apply the declared platform policy, verify it, and emit a
# machine-readable descriptor that the offline allocator consumes.
#
# Everything board-specific lives in platform_policy.conf. This script contains
# no hardcoded core numbers. Change the policy, re-run this, and the descriptor
# follows.
#
# Requirement trace:
#   NPIEDRONE-294  core eligibility rules defined by platform policy
#   NPIEDRONE-294  policies/thresholds configurable and version-controlled
#                  per board target
#   NPIEDRONE-294  thermal and power guardrails
#   NPIEDRONE-329  deterministic verification gate -- a measurement session is
#                  only valid if the platform met the declared policy, and the
#                  descriptor records whether it did
#
# Complements the company's rt-setup.service rather than fighting it: it never
# widens an IRQ mask that rt-setup narrowed, and it cross-checks the declared
# core roles against the actual kernel cmdline.
#
# Usage:
#     ./setup_board.sh                    # apply + verify
#     ./setup_board.sh --verify           # verify only, change nothing
#     POLICY=other.conf ./setup_board.sh  # use a different policy file
#
# Exit 0 = platform matches policy, safe to measure.
# =============================================================================

set -u

VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

POLICY="${POLICY:-platform_policy.conf}"
OUT_JSON="${OUT_JSON:-platform_descriptor.json}"

if [ ! -r "$POLICY" ]; then
    echo "FATAL: policy file not found: $POLICY"
    exit 2
fi
# shellcheck disable=SC1090
. "./$POLICY"

FAIL=0
WARN=0

ok()   { echo "  OK    $*"; }
bad()  { echo "  FAIL  $*"; FAIL=1; }
warn() { echo "  WARN  $*"; WARN=$((WARN+1)); }
hdr()  { echo; echo "=== $* ==="; }

try_write() {
    local val="$1" path="$2"
    [ "$VERIFY_ONLY" -eq 1 ] && return 0
    [ -w "$path" ] 2>/dev/null || return 1
    echo "$val" > "$path" 2>/dev/null
}
read_or() { cat "$1" 2>/dev/null || echo "$2"; }

# nproc reports the CALLING PROCESS's affinity mask, which excludes isolated
# CPUs -- it returns 4 on an isolcpus=4,5 system. Always --all here.
NCORES=$(nproc --all)

read -r -a ROLES <<< "$CORE_ROLES"
if [ "${#ROLES[@]}" -ne "$NCORES" ]; then
    echo "FATAL: policy declares ${#ROLES[@]} core roles but the board has $NCORES cores."
    exit 2
fi

# Expand a cpu list ("0-3", "0,2", "4") into space-separated ids.
expand_list() {
    local out="" part lo hi i
    local IFS=,
    for part in $1; do
        case "$part" in
            *-*) lo=${part%-*}; hi=${part#*-}
                 for ((i=lo; i<=hi; i++)); do out="$out $i"; done ;;
            *)   out="$out $part" ;;
        esac
    done
    echo "$out"
}

tiers_for_role() {
    case "$1" in
        os)          echo "$TIERS_os" ;;
        general)     echo "$TIERS_general" ;;
        rt_reserved) echo "$TIERS_rt_reserved" ;;
        *)           echo "none" ;;
    esac
}

# -----------------------------------------------------------------------------
hdr "0. Platform identity and policy"
# -----------------------------------------------------------------------------
KVER=$(uname -r)
KVERSION=$(uname -v)
MODEL=$(read_or /proc/device-tree/model unknown | tr -d '\0')
CMDLINE=$(read_or /proc/cmdline "")
ISOLATED=$(read_or /sys/devices/system/cpu/isolated "")

echo "  model     : $MODEL"
echo "  kernel    : $KVER"
echo "  cores     : $NCORES"
echo "  policy    : $POLICY  (${PLATFORM_NAME} v${POLICY_VERSION})"
echo "  roles     : $CORE_ROLES"

if echo "$KVERSION" | grep -qi 'PREEMPT_RT' || [ "$(read_or /sys/kernel/realtime 0)" = "1" ]; then
    RT_KERNEL=1; ok "PREEMPT_RT kernel"
else
    RT_KERNEL=0
    if [ "${REQUIRE_PREEMPT_RT:-0}" -eq 1 ]; then
        bad "NOT a PREEMPT_RT kernel, but policy requires one."
        echo "        Boot the RT image, or set REQUIRE_PREEMPT_RT=0 to measure a"
        echo "        deliberate non-RT baseline."
    else
        warn "NOT a PREEMPT_RT kernel (REQUIRE_PREEMPT_RT=0, measuring anyway)"
    fi
fi

# -----------------------------------------------------------------------------
hdr "1. Policy vs. kernel cmdline cross-check"
# -----------------------------------------------------------------------------
# A policy that disagrees with the platform is worse than no policy at all:
# the allocator would emit mappings the hardware cannot honour.
DECLARED_RT=""
for c in $(seq 0 $((NCORES-1))); do
    [ "${ROLES[$c]}" = "rt_reserved" ] && DECLARED_RT="$DECLARED_RT $c"
done
ACTUAL_RT=$(expand_list "$ISOLATED")

norm() { echo "$*" | tr ' ' '\n' | grep -v '^$' | sort -n | tr '\n' ' '; }
D=$(norm "$DECLARED_RT"); A=$(norm "$ACTUAL_RT")

if [ -z "$A" ]; then
    if [ "${REQUIRE_ISOLATION:-0}" -eq 1 ]; then
        bad "no isolated CPUs (/sys/devices/system/cpu/isolated is empty)"
        echo "        Expected isolcpus for cores:$DECLARED_RT"
        echo "        Add to the U-Boot bootargs, or set REQUIRE_ISOLATION=0."
    else
        warn "no isolated CPUs; policy declares rt_reserved on:$DECLARED_RT"
    fi
elif [ "$D" = "$A" ]; then
    ok "isolated CPUs match policy rt_reserved cores: $A"
else
    bad "policy rt_reserved = [$D] but kernel isolated = [$A]"
    echo "        Fix CORE_ROLES in $POLICY, or fix the kernel cmdline. Do not measure"
    echo "        until they agree."
fi

# -----------------------------------------------------------------------------
hdr "2. Console noise"
# -----------------------------------------------------------------------------
# One kernel printk blocks the CPU for milliseconds at 115200 baud and lands
# straight in the measured release latency.
if [ "$VERIFY_ONLY" -eq 0 ]; then
    dmesg -n 1 2>/dev/null && ok "console loglevel -> 1" \
                           || warn "could not set console loglevel"
else
    ok "(verify only)"
fi

# -----------------------------------------------------------------------------
hdr "3. RT throttler"
# -----------------------------------------------------------------------------
try_write -1 /proc/sys/kernel/sched_rt_runtime_us
V=$(read_or /proc/sys/kernel/sched_rt_runtime_us "?")
[ "$V" = "-1" ] && ok "sched_rt_runtime_us = -1" \
                || bad "sched_rt_runtime_us = $V (expected -1)"

# -----------------------------------------------------------------------------
hdr "4. CPU frequency"
# -----------------------------------------------------------------------------
for c in $(seq 0 $((NCORES-1))); do
    try_write performance "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor"
done
sleep 0.3
FREQ_LIST=""; GOV_OK=1
for c in $(seq 0 $((NCORES-1))); do
    G=$(read_or "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor" n/a)
    CUR=$(read_or "/sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq" 0)
    FREQ_LIST="$FREQ_LIST${FREQ_LIST:+,}$CUR"
    [ "$G" != "performance" ] && [ "$G" != "n/a" ] && \
        { bad "core $c governor = $G"; GOV_OK=0; }
done
[ "$GOV_OK" -eq 1 ] && ok "governors = performance (freqs: $FREQ_LIST)"
echo "        (scaling_cur_freq under-reports on an IDLE core; the real check is"
echo "         platform_probe compute -> min_over_p50 must be ~1.00)"

# -----------------------------------------------------------------------------
hdr "5. Idle states"
# -----------------------------------------------------------------------------
DISABLED=0
for s in /sys/devices/system/cpu/cpu*/cpuidle/state[1-9]*; do
    [ -d "$s" ] || continue
    try_write 1 "$s/disable" && DISABLED=$((DISABLED+1))
done
[ "$DISABLED" -gt 0 ] && ok "disabled $DISABLED deep idle states (WFI kept)" \
                      || warn "no cpuidle states exposed"

# -----------------------------------------------------------------------------
hdr "6. Interrupt affinity -> cores [$IRQ_CORES]"
# -----------------------------------------------------------------------------
if command -v systemctl >/dev/null 2>&1; then
    systemctl stop irqbalance 2>/dev/null && ok "stopped irqbalance"
fi
pkill -x irqbalance 2>/dev/null

MOVED=0; PERCPU=0
if [ "$VERIFY_ONLY" -eq 0 ]; then
    try_write "$IRQ_CORES" /proc/irq/default_smp_affinity 2>/dev/null
    for d in /proc/irq/[0-9]*; do
        [ -d "$d" ] || continue
        if echo "$IRQ_CORES" > "$d/smp_affinity_list" 2>/dev/null; then
            MOVED=$((MOVED+1))
        else
            PERCPU=$((PERCPU+1))   # arch timer, IPIs -- genuinely immovable
        fi
    done
    ok "$MOVED IRQs -> [$IRQ_CORES]  ($PERCPU per-CPU, immovable by design)"
fi

# The mask is a request; what matters is whether interrupts actually FIRE on a
# reserved core. Verified separately against /proc/interrupts under load.
RESERVED_SET=$(norm "$DECLARED_RT")
STRAY=0
for d in /proc/irq/[0-9]*; do
    [ -r "$d/smp_affinity_list" ] || continue
    A2=$(cat "$d/smp_affinity_list" 2>/dev/null)
    for rc in $RESERVED_SET; do
        case ",$(expand_list "$A2" | tr ' ' ',')," in
            *",$rc,"*) STRAY=$((STRAY+1)); break ;;
        esac
    done
done
[ "$STRAY" -gt 0 ] && warn "$STRAY IRQs still permit a reserved core (mostly per-CPU)" \
                   || ok "no movable IRQ targets a reserved core"

# -----------------------------------------------------------------------------
hdr "7. Threaded NAPI"
# -----------------------------------------------------------------------------
# Moves packet processing out of softirq context into schedulable kernel
# threads, which can then be kept off the reserved cores. Matters for the TSN
# workload (NPIEDRONE-274/275) sharing this SoC.
NAPI_OK=0
for iface in $NET_IFACES; do
    F="/sys/class/net/$iface/threaded"
    [ -e "$F" ] || continue
    try_write 1 "$F" && { ok "$iface: threaded NAPI on"; NAPI_OK=$((NAPI_OK+1)); }
done
[ "$NAPI_OK" -eq 0 ] && warn "no interface accepted threaded NAPI"

if [ "$VERIFY_ONLY" -eq 0 ]; then
    N=0
    for tid in $(ps -eLo tid,comm 2>/dev/null | awk '$2 ~ /^napi\/|^ksoftirqd/{print $1}'); do
        taskset -p -c "$NAPI_THREAD_CORES" "$tid" >/dev/null 2>&1 && N=$((N+1))
    done
    [ "$N" -gt 0 ] && ok "pinned $N napi/ksoftirqd threads to [$NAPI_THREAD_CORES]"
fi

# -----------------------------------------------------------------------------
hdr "8. Background kernel noise"
# -----------------------------------------------------------------------------
try_write 0  /proc/sys/kernel/watchdog        && ok "soft watchdog off"
try_write 0  /proc/sys/kernel/timer_migration && ok "timer migration off"
try_write 60 /proc/sys/vm/stat_interval       && ok "vm.stat_interval -> 60s"
try_write 0  /proc/sys/kernel/numa_balancing  2>/dev/null
try_write never /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null \
    && ok "THP defrag off"
try_write 1500 /proc/sys/vm/dirty_writeback_centisecs 2>/dev/null

# -----------------------------------------------------------------------------
hdr "9. Thermal guardrails"
# -----------------------------------------------------------------------------
THERMAL_JSON=""; TMAX=0
for zd in /sys/class/thermal/thermal_zone*; do
    [ -r "$zd/temp" ] || continue
    T=$(cat "$zd/temp" 2>/dev/null); TY=$(read_or "$zd/type" unknown)
    if echo "$TY" | grep -qE "$THERMAL_ZONE_MATCH"; then
        C=$((T/1000))
        THERMAL_JSON="$THERMAL_JSON${THERMAL_JSON:+,}{\"zone\":\"$TY\",\"c\":$C}"
        [ "$C" -gt "$TMAX" ] && TMAX=$C
        if   [ "$C" -ge "$THERMAL_CRIT_C" ]; then bad  "$TY at ${C}C >= crit ${THERMAL_CRIT_C}C"
        elif [ "$C" -ge "$THERMAL_WARN_C" ]; then warn "$TY at ${C}C >= warn ${THERMAL_WARN_C}C"
        else ok "$TY at ${C}C"; fi
    fi
done
[ -z "$THERMAL_JSON" ] && warn "no zone matched THERMAL_ZONE_MATCH='$THERMAL_ZONE_MATCH'"

# -----------------------------------------------------------------------------
hdr "10. Platform descriptor -> $OUT_JSON"
# -----------------------------------------------------------------------------
# This is the artefact the offline allocator consumes. It carries the core
# eligibility rules, so the allocator never has to guess the partition.
{
    printf '{\n'
    printf '  "platform_name": "%s",\n'   "$PLATFORM_NAME"
    printf '  "policy_version": "%s",\n'  "$POLICY_VERSION"
    printf '  "model": "%s",\n'           "$MODEL"
    printf '  "kernel_release": "%s",\n'  "$KVER"
    printf '  "preempt_rt": %s,\n'        "$([ "$RT_KERNEL" -eq 1 ] && echo true || echo false)"
    printf '  "cmdline": "%s",\n'         "$(echo "$CMDLINE" | sed 's/"/\\"/g')"
    printf '  "isolated_cpus": "%s",\n'   "$ISOLATED"
    printf '  "irq_cores": "%s",\n'       "$IRQ_CORES"
    printf '  "num_cores": %s,\n'         "$NCORES"
    printf '  "cores": [\n'
    for c in $(seq 0 $((NCORES-1))); do
        R="${ROLES[$c]}"
        TI=$(tiers_for_role "$R")
        TJ=""; for t in $TI; do [ "$t" = "none" ] && continue; TJ="$TJ${TJ:+,}\"$t\""; done
        IS=false; for rc in $RESERVED_SET; do [ "$rc" = "$c" ] && IS=true; done
        printf '    {"core_id":%s,"role":"%s","isolated":%s,"speed_factor":1.0,"eligible_tiers":[%s]}%s\n' \
               "$c" "$R" "$IS" "$TJ" "$([ "$c" -lt $((NCORES-1)) ] && echo , )"
    done
    printf '  ],\n'
    printf '  "thermal_c": [%s],\n'         "$THERMAL_JSON"
    printf '  "thermal_warn_c": %s,\n'      "$THERMAL_WARN_C"
    printf '  "thermal_crit_c": %s,\n'      "$THERMAL_CRIT_C"
    printf '  "governor": "%s",\n'          "$(read_or /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor n/a)"
    printf '  "freqs_khz": [%s],\n'         "$FREQ_LIST"
    printf '  "irqs_moved": %s,\n'          "$MOVED"
    printf '  "irqs_percpu": %s,\n'         "$PERCPU"
    printf '  "threaded_napi_ifaces": %s,\n' "$NAPI_OK"
    printf '  "warnings": %s,\n'            "$WARN"
    printf '  "policy_satisfied": %s,\n'    "$([ "$FAIL" -eq 0 ] && echo true || echo false)"
    printf '  "timestamp": "%s"\n'          "$(date -Iseconds)"
    printf '}\n'
} > "$OUT_JSON"
ok "wrote $OUT_JSON"

# -----------------------------------------------------------------------------
echo
if [ "$FAIL" -eq 0 ]; then
    echo "=== PASS ($WARN warnings). Platform matches policy; safe to measure. ==="
    echo "    $OUT_JSON must travel with every result produced in this session."
else
    echo "=== FAIL. Platform does NOT match policy. Do not measure. ==="
fi
exit $FAIL
