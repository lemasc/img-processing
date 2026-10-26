#!/usr/bin/env bash
# Run a benchmark with controlled conditions: memory-capped, no CPU throttling, pinned to P-threads.
# Usage: scripts/bench.sh [-t single|cores|multi|all] [-m MEM] <command...>
#   single  cpu 0           (one P-thread; keep cpu 1, its HT sibling, idle)
#   cores   cpus 0,2,..,10  (6 P-cores, one thread each, no HT)
#   multi   cpus 0-11       (6 P-cores with HT)  [default]
#   all     cpus 0-15       (P + E cores, for the E-core comparison)
# Exports BENCH_CPUS and OMP_NUM_THREADS/OMP_PLACES for the child.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mode=multi mem=10G
while getopts "t:m:" o; do
  case $o in t) mode=$OPTARG;; m) mem=$OPTARG;; *) exit 2;; esac
done
shift $((OPTIND - 1))
[[ $# -gt 0 ]] || { echo "usage: $0 [-t single|cores|multi|all] [-m MEM] cmd..." >&2; exit 2; }

case $mode in
  single) cpus=0;    threads=1;;
  cores)  cpus=0,2,4,6,8,10; threads=6;;
  multi)  cpus=0-11; threads=12;;
  all)    cpus=0-15; threads=16;;
  *) echo "bad mode: $mode" >&2; exit 2;;
esac

gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
[[ $gov == performance ]] || echo "WARNING: CPU governor is '$gov', not 'performance'" >&2
load=$(cut -d' ' -f1 /proc/loadavg)
awk -v l="$load" 'BEGIN{exit !(l>1.0)}' && echo "WARNING: load average $load, machine is not quiet" >&2

export BENCH_CPUS=$cpus BENCH_MODE=$mode OMP_NUM_THREADS=$threads OMP_PLACES=threads OMP_PROC_BIND=close
echo "bench: mode=$mode cpus=$cpus threads=$threads governor=$gov load=$load" >&2
exec "$here/run-capped.sh" -m "$mem" -q none -n 0 -c "$cpus" "$@"
