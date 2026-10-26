#!/usr/bin/env bash
# Run a command in a resource-capped cgroup scope so a runaway job cannot take down the machine.
# Usage: scripts/run-capped.sh [-m MEM] [-q CPUQUOTA|none] [-n NICE] [-c CPULIST] <command...>
#   -m  memory cap, no swap (default 10G)
#   -q  CPU quota, e.g. 1200% = 12 threads (default 1200%, "none" disables)
#   -n  nice level (default 10)
#   -c  pin to CPU list via taskset, e.g. 0-11 (P-threads). Default: no pinning.
# Note: cpuset is not delegated to user units here, so pinning uses taskset, not AllowedCPUs.
set -euo pipefail
mem=10G quota=1200% nice=10 cpus=
while getopts "m:q:n:c:" o; do
  case $o in m) mem=$OPTARG;; q) quota=$OPTARG;; n) nice=$OPTARG;; c) cpus=$OPTARG;; *) exit 2;; esac
done
shift $((OPTIND - 1))
[[ $# -gt 0 ]] || { echo "usage: $0 [-m MEM] [-q QUOTA|none] [-n NICE] [-c CPULIST] cmd..." >&2; exit 2; }

props=(-p MemoryMax="$mem" -p MemorySwapMax=0)
[[ $quota == none ]] || props+=(-p CPUQuota="$quota")
[[ -z $cpus ]] || set -- taskset -c "$cpus" "$@"
set -- nice -n "$nice" "$@"
exec systemd-run --user --scope -q "${props[@]}" "$@"
