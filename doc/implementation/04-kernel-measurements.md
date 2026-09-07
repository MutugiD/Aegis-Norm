# F04 measurement foundation

Status: initial T4 execution and raw-data review passed; see the [F04 evidence report](../evaluation/08-f04-measurement-review.md). This is the first measurement increment supporting F04 and a provisional part of F09, not completed kernel optimization or the full release evaluation harness. Requirements: REQ-K07, REQ-E01, REQ-E02; initial supporting coverage for T-E01, T-E02 and T-E07.

## What this increment does

The existing scalar kernel remains unchanged. The experiment compares its raw registered operator with the cast-ordered eager expression on identical seeded inputs. The public strict dispatcher is checked before timing; the raw native call cannot fall back. Both timed arms allocate outputs. Python wrapper validation is outside the measured region, while native C++ validation and dispatch remain included.

The related `torch.nn.functional.rms_norm` implementation is executed with explicit epsilon and its numerical differences recorded. It is not silently treated as an identical casting baseline and is not timed in this increment. Compiled eager and existing fused implementation timing/availability reports remain F09 work. This bounded experiment cannot establish superiority over all optimized alternatives.

## The measurement sequence

1. Record a clean source commit, actual preflight, installed package source hashes and dependency versions. Build/load the extension explicitly on the notebook host, recording build/load time separately.
2. Generate one input and weight using a CUDA generator seeded with 2026 plus the case index. Reuse them for both arms and every trial: this is a warm-cache experiment.
3. Compare native output against the cast-ordered reference using the established FP16/FP32 tolerances. Export errors before timing; a failed numerical gate stops the run.
4. Warm each arm for 100 calls. Measure 20 calibration calls and choose one repetition count for both arms, targeting 10 ms for the faster arm, bounded to 20..10000. Record calibration separately. Count measured groups falling below the target; do not hide them.
5. Collect 30 paired trials. A seeded AB/BA order varies which arm runs first. Each sample contains total event duration, synchronized host duration and repetitions.
6. Save each sample and completed case before continuing. On handled failure or interruption, retain partial observations and write a terminal manifest and checksums. A provider forcibly terminating the process may leave a `running` manifest; that snapshot is incomplete and must not be reported as completed.
7. Summarize each case separately, retaining every observation. No pooling across shapes, outlier removal or model-speedup inference.

## What the numbers mean

`microseconds_per_call = total_milliseconds * 1000 / repetitions`.

CUDA events bracket the repeated calls on the current stream; synchronizing the end event establishes completion before reading elapsed time. Event resources are initialized before the measured region. The event interval can include Python submission gaps. The monotonic host interval includes submission and end-event synchronization. Neither is labeled pure hardware kernel duration.

`speedup = median(reference_time) / median(native_time)`; a value above one favors native. Resample whole paired trial indices 10000 times using seed 2026, then take the 2.5th/97.5th percentiles of those median ratios. An interval wholly above one is a preliminary per-case improvement, wholly below one a regression, otherwise inconclusive. The bundle still requires human evidence review. An interval does not prove independent reproducibility.

The summary reports count, median, p95, min/max and IQR for event and host times separately. The six-case smoke profile and nonstandard trial counts are setup validation, not headline performance evidence. The required profile has 72 shape/dtype cases; scalar safety has 16 offset cases. Shapes and bounds follow the [evaluation protocol](../evaluation/01-protocol.md).

## Artifacts and scope

Run [the F04 notebook](../../notebooks/04-t4-kernel-measurements.ipynb) from top to bottom with its full feature commit. It reruns numerical and registration suites, checks the timing path, then runs the required and scalar-safety profiles. Download `aegis-f04-evidence.zip` even after a failed cell.

Each measurement run has a unique directory containing `manifest.json`, `environment.txt`, `correctness.jsonl`, `related-baseline.jsonl`, `calibration.jsonl`, `samples.jsonl`, `summary.json` and `checksums.sha256` for every payload including the finalized manifest. Missing files on a failed run represent steps not reached. Completed status only describes this increment's scope. The provisional `f04-baseline-v1` manifest is not the complete F09 artifact schema: compiler/third-party timings, GPU UUID/clock telemetry, memory/profiler observations, dependency-lock qualification and external artifact validation remain outstanding. No measured DRAM-bandwidth or memory-saving claim is emitted.

Notebook compiler output and JUnit files remain in the outer evidence archive. After reviewing the initial timing run, select one kernel geometry experiment or retain the existing baseline if evidence does not justify a change. Any subsequent kernel change needs paired measurements and full correctness regression.

References: [CUDA event API](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html), [native PyTorch RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.rms_norm.html), and the project's [numerical test contract](../testing/01-test-plan.md).
