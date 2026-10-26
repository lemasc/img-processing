# HCT Assignment: Image Color Processing Across Parallel Platforms

See `docs/plans/full-project-plan.md` for the full plan.

## Build and run
```
uv sync                                  # Python tools (charts, test images)
scripts/env_report.sh                    # record versions to results/env.txt
cmake -S . -B build -G Ninja
scripts/run-capped.sh cmake --build build -j4
scripts/run-capped.sh build/smoke
```
`run-capped.sh` limits memory (default 10G, no swap) so an OOM kills the job, not the machine.
