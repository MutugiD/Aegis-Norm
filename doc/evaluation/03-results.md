# Evaluation results and qualification status

Status: initial operator timings measured; model/serving impact and traffic remain **not yet measured**. Updated: 2026-09-07. This document is intentionally populated with honest status, not synthetic benchmark numbers.

| Experiment | Execution status | Result | Evidence |
|---|---|---|---|
| EXP-ENV: T4 build/preflight | Passed initial run | Fresh compiler logs, metadata, inventory and verified hashes; 38 foundation tests passed | [F02 review](06-f02-gpu-evidence-review.md) |
| EXP-NUM: native correctness | Passed initial suite | 223 tests passed; no failures/errors/skips; sanitizer coverage pending | [F02 review](06-f02-gpu-evidence-review.md) |
| EXP-REG: framework registration | Passed initial suite | 10 native framework and 32 metadata cases passed; 223 numerical regressions passed | [F03 review](07-f03-gpu-evidence-review.md) |
| EXP-KERNEL: operator latency | Preliminary single-session run | 88 cases, 5280 raw samples; eager/native comparisons only | [F04 review](08-f04-measurement-review.md) |
| EXP-KERNEL: device-memory traffic | Not run | Not yet measured | No profiling |
| EXP-MODEL: numeric/generation/performance | Not run | Not yet measured | No model execution |
| EXP-SERVE: client experience | Not run | Not yet measured | No server implementation or endpoint |
| EXP-REPRO: independent contributor | Not run | Not yet measured | No contributor artifacts |

The C2 compatibility candidate is not a passing support matrix. Contributor setup logs are partial environment evidence, not GPU execution qualification. Documentation validation, source inspection and algebraic memory estimates are not GPU qualification. Only the scoped operator comparisons in the F04 report are established. No model speedup, TTFT reduction, VRAM saving, serving throughput increase or operational cost reduction is established.

## Per-run report to add after execution

Record run ID and immutable artifact location; scope and actual environment; correctness outcome; baseline compatibility; per-shape/per-workload timings with variability; paired intervals; failures and omitted optional observations; comparison with previous runs; and the precise claim supported. Link every table value to a raw case/arm/trial set. Mark draft/preliminary versus independently reproduced conclusions.

## Release decision

Current decision: documentation available for review; software release unqualified. The first release needs the required functionality, numerical/safety checks, T4 experiments and reproduction evidence. An eventual finding of negligible end-to-end speedup must be stated directly and used to prioritize later work.
