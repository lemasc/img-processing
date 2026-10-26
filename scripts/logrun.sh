#!/usr/bin/env bash
# Run a command and save its command line, timestamp, and output to results/logs/<name>.txt
# for the report. Usage: scripts/logrun.sh <name> <command...>
set -uo pipefail
name=$1; shift
log=results/logs/$name.txt
mkdir -p results/logs
{
  echo "\$ $*"
  echo "# $(date -Is)  host=$(hostname)  commit=$(git rev-parse --short HEAD 2>/dev/null)"
  "$@" 2>&1
  echo "# exit=$?"
} | tee "$log"
