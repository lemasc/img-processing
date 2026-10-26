#!/usr/bin/env bash
# Record hardware and software versions for the report.
out=${1:-results/env.txt}
mkdir -p "$(dirname "$out")"
{
  echo "date: $(date -Is)"
  echo "== GPU"
  nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,clocks.max.mem --format=csv
  echo "== CUDA"; nvcc --version | tail -2
  echo "== Compilers"; gcc --version | head -1; cmake --version | head -1; ninja --version
  echo "== Runtimes"; node --version; uv --version; command -v emcc >/dev/null && emcc --version | head -1 || echo "emcc: not installed"
  echo "== Profilers"; ncu --version | tail -1; nsys --version
  echo "== CPU"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|L2|L3"
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
  echo "== Memory"; free -h | head -2
  echo "== Kernel"; uname -sr
} > "$out"
echo "wrote $out"
