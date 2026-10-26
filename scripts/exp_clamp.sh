#!/usr/bin/env bash
# Build and run the clamp-branch experiment on flat, photo, and noise 24 MP images.
# Needs build/libimage_io.a (cmake --build build). Usage: scripts/logrun.sh exp_clamp scripts/exp_clamp.sh
set -euo pipefail
cd "$(dirname "$0")/.."
out=build/exp; mkdir -p $out
base="-O2 -fno-tree-vectorize -fno-tree-slp-vectorize -Iinclude -Iinclude/third_party"
fm="-ffinite-math-only -fno-signed-zeros"
g++ $base -DVARIANT=0 experiments/clamp_branches.cpp build/libimage_io.a -o $out/v0_branchy
g++ $base $fm -DVARIANT=0 experiments/clamp_branches.cpp build/libimage_io.a -o $out/v0_finite_math
g++ $base -DVARIANT=1 experiments/clamp_branches.cpp build/libimage_io.a -o $out/v1_fminf
g++ $base -DVARIANT=3 experiments/clamp_branches.cpp build/libimage_io.a -o $out/v3_std_minmax
g++ $base -DVARIANT=4 experiments/clamp_branches.cpp build/libimage_io.a -o $out/v4_sse
for v in v0_branchy v1_fminf v3_std_minmax v4_sse v0_finite_math; do
  echo "== $v   (minss+maxss in loop: $(objdump -d --no-show-raw-insn $out/$v | awk '/_M_invoke[^>]*>:/,/ret/' | grep -c -E 'minss|maxss'))"
  scripts/bench.sh -t single $out/$v images/flat_24mp.png images/photo_24mp.png images/noise_24mp.png 2>&1 | grep variant
done
