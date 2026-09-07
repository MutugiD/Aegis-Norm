# F04 operator measurement review

Status: preliminary single-session measurements reviewed, 2026-09-07. Tested commit `9a6dcc9ccdfcdeeb08c40a3988dfc655870508db`. [Original archive](runs/f04-20260907/evidence.zip) and [review manifest](runs/f04-20260907/review.json).

## Integrity and correctness

All 305 tests passed: 38 foundation, 223 RMSNorm, 42 framework/metadata and two measurement-path tests; no failures, errors or skips. Both native compiler logs contain actual sm_75 compilation and linking. All 36 top-level artifact hashes, five nested smoke hashes and 14 nested measurement hashes verify; these indexes overlap rather than representing 55 distinct files. All 42 source-hash checks across smoke and two measurement manifests match the tested Git blobs.

Recomputed every per-case summary, including all 10000-resample paired bootstrap intervals, directly from the raw samples. All 88 summaries match exactly. Both profiles completed with clean worktrees; all 88 numerical gates passed. The related native PyTorch RMSNorm outputs also passed the selected tolerances, which does not establish identical casting semantics.

Environment: T4, Python 3.13.15, torch 2.14.0+cu126, toolkit 12.6.85, GCC 11.4.0 and driver 580.82.07. Dependency audits found no known vulnerabilities among audited packages; the unpublished project and local torch wheel label remain installed-audit exclusions, with canonical torch covered by the direct audit.

## Observed operator comparisons

Required run `20260907T101133Z-7fbd21e5`: 72 cases and 4320 raw samples. Safety run `20260907T101505Z-033f0bfa`: 16 cases and 960 samples. Each case has 30 paired trials. Every event and host interval favors native over this eager expression. Event ratios range from 2.216 to 8.850 in the required matrix and 5.792 to 8.378 in safety cases. Do not average these ranges into a product-wide speedup.

| Required FP16 case | Eager median us/call | Native median us/call | Event ratio |
|---|---|---|---|
| Rows 1, H 2048 | 88.354 | 10.337 | 8.548 |
| Rows 1, H 4096 | 117.071 | 16.094 | 7.274 |
| Rows 32, H 4096 | 109.137 | 13.973 | 7.810 |
| Rows 2048, H 4096 | 1337.965 | 217.181 | 6.161 |

Values resolve to the named run's `summary.json` and `samples.jsonl` case IDs. These are allocated-output, warm-cache operator timings. Python submission gaps can be substantial, so these are not pure CUDA kernel durations and do not imply a model speedup.

## Calibration and limits

1901 of 5280 groups fell below the calibration target of 10 ms. Minimum groups were 5.042 ms (required) and 3.056 ms (safety). Counts were frozen per case and no observations were removed. The target is a calibration objective, not evidence that every observed duration meets it; future experiments should retain the same diagnostics and account for drift.

No clock/traffic profiling, independently repeated run, compiled eager timing or third-party fused timing is included. No performance claim beyond this eager/native comparison is justified. Full F09 conformance, sanitizer qualification and model/serving evaluations remain outstanding.

## Next experiment

Compare a 128-thread-per-row candidate directly against the retained 256-thread native kernel. Fewer warps reduce the number of shared partial sums but increase per-thread work; occupancy and reduction-order effects require measurement. Preserve the default operator and casting boundary, run the full numerical suite for both variants, and collect paired native/native observations before deciding whether any dispatch change is warranted. The current results alone do not justify promoting a new geometry.
