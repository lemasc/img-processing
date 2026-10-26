#!/usr/bin/env bash
# Measure achievable ceilings for the roofline: host DRAM (1 and 12 threads), PCIe, link speed.
# Usage: scripts/logrun.sh exp_peaks scripts/exp_peaks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/exp
g++ -O3 -march=native -fopenmp experiments/host_bandwidth.cpp -o build/exp/host_bandwidth
nvcc -O2 -arch=sm_86 experiments/pcie_bandwidth.cu -o build/exp/pcie_bandwidth
nvcc -O2 -arch=sm_86 experiments/pcie_duplex.cu -o build/exp/pcie_duplex
echo "== host DRAM (theoretical 76.8 GB/s)"
scripts/bench.sh -t single build/exp/host_bandwidth 2>&1 | grep threads
scripts/bench.sh -t multi build/exp/host_bandwidth 2>&1 | grep threads
echo "== PCIe"
( sleep 2; echo "link speed during run: $(cat /sys/bus/pci/devices/0000:01:00.0/current_link_speed), width $(cat /sys/bus/pci/devices/0000:01:00.0/current_link_width)" ) &
scripts/bench.sh -t single build/exp/pcie_bandwidth 2>&1 | grep -v -E "^bench:"
wait
echo "== PCIe, both directions at once (Stage 11)"
scripts/bench.sh -t single build/exp/pcie_duplex 2>&1 | grep -v -E "^bench:"
