#!/usr/bin/env bash
# Explain the 1-level CPU/GPU differences: compare tolerance-0 results with FMA fusion on and off.
# Usage: scripts/logrun.sh exp_fma scripts/exp_fma.sh
set -uo pipefail
cd "$(dirname "$0")/.."
imgs="images/photos/24mp_night_shot.jpg images/noise_24mp.png"
cmake -S . -B build -G Ninja -DCOLOR_CUDA_FMAD=ON >/dev/null && cmake --build build --target check_color >/dev/null
cmake -S . -B build-nofma -G Ninja -DCOLOR_CUDA_FMAD=OFF >/dev/null && cmake --build build-nofma --target check_color >/dev/null
for cfg in build build-nofma; do
  echo "=== $cfg (tolerance 0)"
  CHECK_IMPL=$cfg scripts/run-capped.sh -m 6G $cfg/check_color --tol 0 $imgs | grep -E "all 16.7M|night|noise|RESULT"
done
