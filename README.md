# HCT Assignment: Image Color Processing Across Parallel Platforms

See `docs/plans/full-project-plan.md` for the full plan.

## Build and run
```
uv sync                                  # Python tools (charts, test images)
scripts/env_report.sh                    # record versions to results/env.txt
cmake -S . -B build -G Ninja
scripts/run-capped.sh cmake --build build -j4
scripts/bench.sh -t single build/smoke   # benchmark run, pinned
```

## Safe running (remote machine)
- `scripts/run-capped.sh [-m MEM] [-q QUOTA|none] [-n NICE] [-c CPULIST] cmd...` runs in a cgroup scope: memory cap with no swap (default 10G), CPU quota (default 1200%) and nice 10. Use it for builds and any other heavy job, so an OOM kills the job and not the machine.
- `scripts/bench.sh [-t single|multi|all] [-m MEM] cmd...` runs benchmarks with no CPU quota, nice 0, pinned to P-threads (`single` = cpu 0, `multi` = 0-11, `all` = 0-15), and exports `BENCH_CPUS`, `OMP_NUM_THREADS` and `OMP_PLACES`. It warns if the governor is not `performance` or the load is above 1.
- Pinning uses `taskset` because cpuset is not delegated to user cgroups on this machine.
