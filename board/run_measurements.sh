#!/bin/sh
# Run the execution-time campaign on the i.MX95 and write the raw results.
#
#   sudo ./run_measurements.sh              all 6 cores, 100k iterations each
#   sudo ./run_measurements.sh 10000        10k iterations, for a quick check
#
# Writes a55_exec_raw.json here. Then, in the repo:
#   python scripts/generator/map_measured_to_catalogue.py
#   python scripts/generator/gen_exec_table.py
#
# Checks the board is in a state worth measuring before it starts. A run on a
# core that is not isolated, or with the governor free to change frequency
# mid-run, produces a distribution that describes the governor rather than the
# code.

set -e

ITERS=${1:-100000}
OUT=a55_exec_raw.json

echo "== board state =="
uname -srm
echo "isolated cpus : $(cat /sys/devices/system/cpu/isolated 2>/dev/null || echo unknown)"
echo "nohz_full     : $(cat /sys/devices/system/cpu/nohz_full 2>/dev/null || echo unknown)"
echo "cmdline       : $(cat /proc/cmdline)"

bad=0
for c in 0 1 2 3 4 5; do
    gov=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null || echo unknown)
    echo "governor cpu$c : $gov"
    [ "$gov" = "performance" ] || bad=1
done

if [ "$bad" = "1" ]; then
    echo
    echo "WARNING: not every core is on the 'performance' governor."
    echo "Frequency changes during the run widen the tail, and the p99.9 over"
    echo "p50 ratio then describes the governor rather than the code. Set it:"
    echo "  for c in 0 1 2 3 4 5; do"
    echo "    echo performance > /sys/devices/system/cpu/cpu\$c/cpufreq/scaling_governor"
    echo "  done"
    echo
    printf "continue anyway? [y/N] "
    read -r ans
    [ "$ans" = "y" ] || exit 1
fi

case " $(cat /sys/devices/system/cpu/isolated 2>/dev/null | tr ',' ' ') " in
    *" 4 "*) : ;;
    *) echo "NOTE: cpu4 is not in the isolated set. The isolated class will"
       echo "measure the same thing as the general class." ;;
esac

echo
echo "== building =="
make -s

echo
echo "== measuring: 25 tasks x 6 cores, $ITERS iterations each =="
echo "   that is 150 runs; at 100k iterations expect a couple of hours"
./measure_wcet --iters "$ITERS" > "$OUT"

echo
echo "wrote $OUT"
echo "tasks measured: $(grep -c '\"nominal_ns\"' "$OUT")"
echo
echo "next, in the repo:"
echo "  cp $OUT <repo>/board/a55_exec_raw.json"
echo "  python scripts/generator/map_measured_to_catalogue.py --raw board/a55_exec_raw.json"
echo "  python scripts/generator/gen_exec_table.py --force"
echo "  python scripts/run_allocators.py       # --force invalidates the old results"
