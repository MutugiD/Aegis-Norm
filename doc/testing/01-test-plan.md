# Correctness, integration and functionality testing

Status: full release protocol specified. F01 reference/preflight/package tests are implemented; the contributor run passed both initial GPU smoke tests; full native RMSNorm tests remain unexecuted. Partial F01 coverage does not complete the native RMSNorm cases below. Use pytest-style automated tests, parameterization and fixed seeds. Do not make CI download model weights for basic CPU tests.

## Numerical contract and data

The authoritative reference is the cast-ordered expression in the [operator contract](../architecture/02-operator-contract.md). Use an additional CPU FP64 mathematical oracle to diagnose reduction error; it does not replace FP32/cast semantics for release acceptance.

For finite ordinary outputs use elementwise `abs(actual-reference) <= atol + rtol*abs(reference)` across every element:

| Native dtype | atol | rtol | Ordinary data |
|---|---|---|---|
| FP32 | 1e-6 | 2e-5 | Seeds 0, 17, 123; normal, uniform [-10,10], constant, alternating signs, scale 1e-3/1/1e3 |
| FP16 | 2e-3 | 2e-3 | Same distributions represented in FP16; gamma uniform [-2,2], zeros, ones |

Test eps 1e-6, 1e-5 and 1e-3. Record max absolute error, max relative error with denominator floor 1e-8, RMS error, failing count and worst index. Relative error near zero is diagnostic; allclose uses the formula above. Do not loosen tolerances after seeing failures without an explicit reviewed contract change and complete rerun.

Special cases: all zeros, tiny/subnormal values, FP16 finite extrema, large FP32 values that overflow sum-of-squares, NaN, positive/negative infinity, and nonfinite gamma. Compare NaN masks and infinity sign masks first, then finite positions. IEEE zero signs are diagnostic, not a release gate. No blanket `equal_nan` setting may hide unexpected NaNs from finite ordinary inputs.

## Kernel test inventory

| Test | Scenario and acceptance |
|---|---|
| T-K01 | Import on CPU without extension; reference works. Strict native fails clearly. C1 GPU build/import/smoke succeeds before timing |
| T-K02 | H = 1, 7, 31, 32, 33, 96, 160, 255, 256, 257, 768, 1023, 1024, 2048, 4096, 5120, 8192, 65536; rows = 1, 3, 32; ranks 1/2/3; both dtypes meet tolerances |
| T-K03 | Distributions/eps/special values above; finite inputs within ordinary range do not create unexpected NaNs |
| T-K04 | Invalid rank, zero H, wrong gamma shape, integer dtype, mismatched device, nonfinite/nonpositive epsilon, unsupported backend and active gradients follow documented errors/fallbacks |
| T-K05 | Empty leading dimension returns empty without launch. Contiguous nonzero offsets exercise unaligned x/gamma; transpose and strided slices exercise reference fallback. Native call rejects noncontiguous inputs |
| T-K06 | Allocate/write/normalize/consume on a side stream with explicit event dependencies and compare after completion. Do not mask a default-stream bug by synchronizing before operator call. Multi-device guard test is optional when two devices exist |
| T-K07 | Verify x/gamma unchanged, output allocation independent, shape/dtype/device correct, repeated calls/lifetime safe. Profile separately to detect unintended synchronization |
| T-K08 | Run compute-sanitizer memcheck, racecheck and synccheck on small/odd/unaligned cases when tooling is available. Release qualification requires a recorded sanitizer run on a qualifying host; unavailable notebook tooling is pending, not pass |
| T-K09 | FakeTensor metadata, dispatcher opcheck under inference, and explicit no-backward behavior. Optional compiled operator smoke reports unsupported backend separately |
| T-K10 | Every optimized variant reruns T-K02..T-K08, then paired kernel benchmarks; no shape class is omitted because it regresses |

The original three-warp bug is represented by H=96; test multiple nonzero values in all portions so omitted terms cannot accidentally pass. Large allocation/grid-limit validation should be tested via metadata/fake validation rather than allocating unsafe giant tensors.

## Integration and environment tests

| Test | Scenario and acceptance |
|---|---|
| T-I01 | Patch qualified random small Llama and pinned TinyLlama; preserve weight Parameter identities, epsilon and state keys; unrelated model remains untouched |
| T-I02 | Repeated patch/restore, strict partial failure, conflicting patch and zero-match model; no partially patched state; different backend requires restore |
| T-I03 | Inference, save_pretrained, fresh reload and re-patch; compare state tensors and teacher-forced outputs; newly loaded model starts unpatched |
| T-I04 | CPU, unsupported class/layout/mixed dtype, train mode and gradient-requiring input use original/reference in auto; strict native fails; original gradients still compute |
| T-N01 | Fresh notebook/environment build with recorded Python, torch runtime, nvcc, compiler, driver and architecture; CPU-only import checked separately |
| T-N02 | Missing GPU/nvcc, unexpected GPU, dependency mismatch, failed model download; actionable preflight outcomes and no false T4 result |
| T-N03 | Interrupt after a benchmark case; export valid completed cases and interrupted manifest; resume uses a new attempt ID without overwriting evidence |
| T-N04 | Notebook-only correctness/model request/export works without tunnel; optional public demo is a separate provider-qualified check |

## API and runtime functionality tests

Use a deterministic fake worker to inject stalls, Unicode chunks and exceptions without consuming GPU quota. Repeat the successful vertical request path against the real model. Use barriers/events or a controllable clock, not brittle fixed sleeps, for queue/deadline tests.

| Test | Scenario and acceptance |
|---|---|
| T-S01 | Malformed JSON, booleans as integers, unknown fields/model, invalid roles/ordering, nonfinite temperature, bad caps and oversized chunked body produce exact status/code |
| T-S02 | Template length includes generation prompt/special tokens; exact context boundary succeeds, one over fails; real nonstream request has correct usage/finish and no prompt echo |
| T-S03 | Hold worker active; admit four queued, reject next; CPU preprocessing is bounded; cancellation/timeout frees a waiting slot; maximum active GPU generation remains one |
| T-S04 | Queued timeout returns 504 before headers; nonstream execution deadline signals stop; failed worker does not leave waiting futures unresolved |
| T-S05 | Loading/warmup failure is not-ready; health responds during generation; token auth for configured endpoint; no sensitive data in error body/logs |
| T-S06 | Parse SSE framing and JSON across arbitrary transport splits; initial role, Unicode/newline content, one finish reason and exactly one DONE on success; ignore heartbeats in output |
| T-S07 | Exception/deadline after headers emits terminal error and DONE, never success; pre-header failure keeps HTTP error status |
| T-S08 | Consumer stalls and fills 64-chunk handoff; producer waits at most 5 seconds before cancellation; no unbounded memory growth or premature active-slot reuse |
| T-S09 | Queued/active disconnect and model stop; worker ignores stop to test 5-second grace and not-ready state; cleanup cannot start another generation on the same active worker |
| T-S10 | Graceful shutdown stops admission, cancels queued work, signals active work; logs distinguish first token from first text; optional tunnel does not alter core output contract |

## Evaluation harness checks

| Test | Scenario and acceptance |
|---|---|
| T-E01 | Artifact validator rejects missing versions, nonfinite timing, negative counts and false completed status; planned null values accepted only for not_run |
| T-E02 | Same seed/tensors/eps/workload in baseline and candidate; strict native dispatch; warmup excluded; units/sample counts correct |
| T-E03 | Teacher-forced corpus and generated-text reports follow quality protocol, including disagreement rather than suppressing it |
| T-E04 | Prefill creates first-token logits; cached decode uses retained state; generated-token denominator excludes prompt tokens; memory counters reset at declared boundary |
| T-E05 | Client TTFT ignores role/heartbeat, counts failures/rejections, differentiates token ITL and text-chunk intervals; local and tunnel runs never pooled |
| T-E06 | Independent run reproduces build/correctness; compare matched environments and statistical uncertainty; disagreement remains visible |
| T-E07 | Report values resolve to raw artifacts and run IDs; no passing/accelerated claims for not_run, skipped or failed records |

## Execution gates

CPU CI runs reference, schema, fake-worker functionality and artifact checks after corresponding code exists. GPU notebook runs correctness before performance. A mandatory failure blocks that configuration's release. Skips list reason, impact and follow-up experiment. Documentation validation checks files/examples/traceability only and must not be reported as any T-* pass.
