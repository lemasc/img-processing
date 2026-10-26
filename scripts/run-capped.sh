#!/usr/bin/env bash
# Run a command inside a memory-capped cgroup so an OOM kills the job, not the machine.
# Usage: scripts/run-capped.sh [-m 10G] <command...>
set -euo pipefail
mem=10G
if [[ "${1:-}" == "-m" ]]; then mem=$2; shift 2; fi
exec systemd-run --user --scope -q -p MemoryMax="$mem" -p MemorySwapMax=0 "$@"
