#!/bin/bash
# =============================================================================
# run_alloc_own.sh -- run the allocation sweep that measures what each task
# OWNS, rather than what its wall clock says.
#
# Needs alloc_own_spec.txt (from `python -m src.task_allocator.export_board_spec`
# on the host) in this directory.
#
# Usage:
#     ./run_alloc_own.sh                 # 5 s per allocation
#     ./run_alloc_own.sh 3               # 3 s per allocation
#
# RESUMABLE. Every allocation writes exactly one line to results_alloc_own.jsonl
# and stdout is flushed after each, so the line count is exactly how far the
# sweep got. Re-running this script counts those lines, tells the tool to skip
# that many, and APPENDS. A 10000-allocation sweep is ~17 hours -- long enough
# that a reboot, a dropped session or a full disk is a real possibility, and
# losing seventeen hours to any of them is not acceptable.
#
# To start over instead of resuming:  rm -rf results/alloc_own
# =============================================================================

set -u

SECS="${1:-5}"
OUT=results/alloc_own
SPEC=alloc_own_spec.txt
TOOL=./measure_alloc_own
JSONL="$OUT/results_alloc_own.jsonl"

mkdir -p "$OUT"

if [ ! -r "$SPEC" ]; then
    echo "ERROR: $SPEC not found. Copy it from the host:"
    echo "  python -m src.task_allocator.export_board_spec --pool all --per-method 2500"
    echo "  -> out/alloc_own_spec.txt"
    exit 1
fi

if [ ! -x "$TOOL" ]; then
    echo "ERROR: $TOOL not built. Run:"
    echo "  gcc -O2 -std=gnu11 -pthread -o measure_alloc_own measure_alloc_own.c statcom_tasks.c -lm -lrt"
    exit 1
fi

N=$(grep -c '^ALLOC ' "$SPEC")

# A kill or a power cut during the write can leave the last line unterminated.
# Drop it before counting -- otherwise the file carries a corrupt record.
if [ -s "$JSONL" ] && [ -n "$(tail -c 1 "$JSONL")" ]; then
    echo "NOTE: last line of $JSONL is incomplete (interrupted write). Dropping it."
    sed '$d' "$JSONL" > "$JSONL.fix" && mv "$JSONL.fix" "$JSONL"
fi

# HOW FAR DID IT GET, IN ALLOCATIONS?
#
# The tool writes ONE LINE PER TASK, not per allocation -- an allocation with
# n_tasks=13 emits 13 lines. The tool's skip argument counts ALLOCATIONS, so
# `wc -l` is the wrong number by a factor of ~16 and would skip most of the
# spec on a resume.
#
# An interrupted run also leaves the last allocation PARTIAL: some of its task
# rows written, the rest not. Resuming past it would leave a half-measured
# allocation in the file forever. So count only allocations whose row count
# reached their own n_tasks, and cut the file at the last complete one. At
# most one allocation is re-measured, which costs six seconds.
DONE=0
if [ -s "$JSONL" ]; then
    set -- $(awk '
        { match($0, /"set_index":[0-9]+/); si = substr($0, RSTART+12, RLENGTH-12)
          match($0, /"n_tasks":[0-9]+/);   nt = substr($0, RSTART+10, RLENGTH-10)
          if (si != cur) { cur = si; want = nt + 0; got = 0 }
          got++
          if (got == want) { complete++; lastline = NR } }
        END { print complete+0, lastline+0 }' "$JSONL")
    DONE=$1
    LASTLINE=$2
    TOTAL=$(wc -l < "$JSONL")
    if [ "$TOTAL" -gt "$LASTLINE" ]; then
        echo "NOTE: last allocation is incomplete ($(( TOTAL - LASTLINE )) stray"
        echo "      task rows). Cutting back to allocation $DONE and re-running it."
        head -n "$LASTLINE" "$JSONL" > "$JSONL.fix" && mv "$JSONL.fix" "$JSONL"
    fi
fi

if [ "$DONE" -ge "$N" ]; then
    echo "Already complete: $DONE/$N allocations in $JSONL"
    echo "To start over:  rm -rf $OUT"
    exit 0
fi

# The spec being resumed must be the spec the earlier session used, or the skip
# count points at the wrong allocations and the file is silently mismatched.
# Cheap to check here, impossible to notice afterwards.
if [ "$DONE" -gt 0 ] && [ -f "$OUT/alloc_own_spec.txt" ]; then
    if ! cmp -s "$SPEC" "$OUT/alloc_own_spec.txt"; then
        echo "ABORT: $SPEC differs from the one this run started with"
        echo "       ($OUT/alloc_own_spec.txt). Resuming would skip the wrong"
        echo "       allocations. Either restore the original spec, or"
        echo "       rm -rf $OUT and start over."
        exit 1
    fi
fi

echo "--- verifying platform state ---"
OUT_JSON="$OUT/platform_descriptor.json" ./setup_board.sh > "$OUT/setup.log" 2>&1 || {
    echo "ABORT: setup_board.sh failed. See $OUT/setup.log"; exit 1; }
echo "  platform OK"

LEFT=$(( N - DONE ))
EST=$(( LEFT * (SECS + 1) ))
echo
echo "Spec:      $SPEC   ($N allocations)"
[ "$DONE" -gt 0 ] && echo "RESUMING:  $DONE already measured, $LEFT to go"
echo "Budget:    ${SECS}s per allocation  (estimated $(( EST / 3600 ))h $(( EST % 3600 / 60 ))m)"
echo "Output:    $JSONL   (appending)"
echo -n "Space:     "; df -h . | tail -1
echo

cp "$SPEC" "$OUT/alloc_own_spec.txt"
touch "$JSONL"

chrt -f 5 $TOOL "$SECS" "$SPEC" "$DONE" \
    >> "$JSONL" 2>>"$OUT/alloc_own.err"

FINAL=$(awk '
    { match($0, /"set_index":[0-9]+/); si = substr($0, RSTART+12, RLENGTH-12)
      match($0, /"n_tasks":[0-9]+/);   nt = substr($0, RSTART+10, RLENGTH-10)
      if (si != cur) { cur = si; want = nt + 0; got = 0 }
      got++
      if (got == want) complete++ }
    END { print complete+0 }' "$JSONL")
echo
echo "========================================================"
if [ "$FINAL" -ge "$N" ]; then
    echo "Done.  $FINAL/$N allocations."
else
    echo "STOPPED EARLY at $FINAL/$N. Nothing is lost -- run this script"
    echo "again with the same spec and it picks up from allocation $FINAL."
fi
[ -s "$OUT/alloc_own.err" ] && { echo; echo "STDERR (last 20):"; tail -20 "$OUT/alloc_own.err"; }
echo
echo "Pull to the host with:"
echo "  tar czf alloc_own_results.tar.gz $OUT"
echo "========================================================"
