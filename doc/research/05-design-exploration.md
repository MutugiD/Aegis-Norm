# Design exploration beyond the collected notes

Status: research-informed proposals and selected experiments, 2026-09-05. The [collected ideas](../product-context.md) are a starting point. This document asks which approaches best serve an observable request path on limited hardware; it does not assume the original kernel or architecture is the answer.

## Opportunity map

| Approach | Potential benefit | Cost / uncertainty | Decision |
|---|---|---|---|
| Correct scalar fused kernel | Small implementation, transparent semantics, robust odd widths | Rereads inputs; launch and allocation may dominate tiny workloads | Required first native baseline |
| Retain per-thread values in registers | Avoid a later input load when registers suffice | Register pressure, spills and occupancy; width specialization | Experiment after scalar baseline, inspect compiler register/spill reports |
| Stage a row in shared memory | Reuse input values explicitly | Shared-memory footprint, barriers, carveout and occupancy | Compare only when profiling indicates traffic is worth trading for capacity |
| Packed FP16 loads/stores | Reduce address/load instructions | Alignment, tail handling and preserving intermediate rounding | Separate optimization PR with scalar fallback |
| Warp-per-row / several rows per block | Improve mapping for narrow rows and many rows | Different reduction and row dispatch; may not help single-row decode | Candidate after required shape matrix identifies a gap |
| Preallocated output / CUDA graphs | Reduce allocation or CPU dispatch overhead | Aliasing/lifetime and graph capture constraints; changes fairness of timing | Optional diagnostic experiment, not hidden inside baseline comparison |
| Existing compiler/kernel integration | Avoid maintaining an inferior bespoke path | Version/hardware limits, cast semantics and integration overhead | Attempt compatible baselines; adopt evidence, not a presumed native advantage |
| Residual-add plus RMSNorm | Remove an additional materialized boundary | Model graph changes and mutation/aliasing semantics | Future feature requiring profile attribution and a new contract |
| Continuous batching | Improve occupancy and amortize per-request overhead | Scheduling fairness, cache budgets and cancellation complexity | Future runtime integration, after serialized serving measurements |

Prior-art evidence and limitations are in [feasibility](01-feasibility.md) and the [source register](04-sources.md). The decisions in the table are project hypotheses and engineering choices, not statements that an alternative will be faster.

## Experiment sequence and selection rules

First reproduce the reference's numeric behavior. Next measure scalar native execution across small decode and larger prefill cases. Change one design dimension at a time while retaining the same workload, data, environment and timing boundaries. Record compilation cost and occupancy/register diagnostics separately from latency.

A candidate can be selected for one shape family and rejected for another. Require passing numerical/safety tests and a paired latency improvement whose interval excludes no change in the selected family. Retain scalar dispatch elsewhere. If the confidence interval overlaps no improvement, label it inconclusive and do not enable it by default merely because its median is lower.

Do not optimize allocation away in only one arm without labeling an operator-contract change. Native operator timing includes its output allocation; a kernel-only preallocated diagnostic is a separate experiment. Likewise, compare cached/uncached inputs and graph replay in separate categories.

## Product hypotheses

**H1: normalization is a material decode cost.** Measure its share on the actual model. If small, the product value may be portability, reproducibility and a reliable integration primitive rather than a large generation speedup.

**H2: device-side savings survive Python integration.** Measure strict native standalone, adapted model and unchanged reference. If host dispatch dominates, investigate the integration boundary before adding more kernel complexity.

**H3: users notice the improvement.** Measure first-content latency and request completion locally before using a tunnel. If queue/network delay dominates, prioritize scheduling or client transport as separate features rather than describing kernel results as user results.

**H4: the project is reproducible on free sessions.** Test a fresh independent notebook. If dependency setup is the primary failure, prioritize environment tooling and artifact diagnostics before expanding the model catalog.

These hypotheses can fail without invalidating the entire project. The release report should explain what was learned, which design choices survived, and which later product work is justified. A supported existing implementation can become the recommended backend if it proves better; maintaining custom CUDA is not an end in itself.
