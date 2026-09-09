# F04 geometry experiment: 128 versus 256 threads

Status: candidate implemented; native compilation, correctness and paired measurements are pending. No candidate promotion or speedup claim. The [initial operator evidence](../evaluation/08-f04-measurement-review.md) motivates direct native/native comparisons instead of another comparison solely against eager execution.

## Hypothesis and mathematics

Both kernels compute the same cast-ordered RMSNorm. A block owns one row, accumulates squared input in FP32, reduces partial sums, computes inverse RMS, rounds normalized values to input dtype, then multiplies by gamma. The candidate uses 128 threads (four warps) instead of 256 (eight warps). Its shared partial-sum array has four entries; the first warp zero-fills its other 28 lanes before reduction. All threads still reach both barriers, including threads with no input elements.

For H=4096, the nominal work per thread increases from 16 to 32 elements. Fewer warp partials may reduce block reduction overhead, while less parallel work and more per-thread accumulation may hurt other shapes. Floating-point summation order changes, so identical mathematics does not guarantee bitwise-identical output. Existing numerical tolerances remain fixed.

This is an empirical hypothesis, not an occupancy guarantee. NVIDIA recommends exploring block sizes while considering resource usage and latency hiding; see [thread/block heuristics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics).

## Isolation and interfaces

The production CUDA and C++ binding files remain byte-for-byte unchanged. Experimental sources live in `csrc/rmsnorm128_cuda.cu` and `csrc/rmsnorm128_binding.cpp`. The validation logic is copied from the retained binding so this experiment does not refactor its baseline; keep contract changes synchronized while the candidate exists.

`aegis_norm.experiments.load_128()` explicitly loads the qualified baseline, then separately compiles/registers `aegis_norm_experiments::rms_norm_128`. It adds matching fake metadata and inference-only Autograd redispatch. Registration is idempotent per process; a partial load failure requires a fresh process. Normal imports, `load_native()` and public auto/cuda dispatch never select or compile this candidate.

## Tests and measurements

The 223-case numerical suite now runs for each operator: 446 cases total. Both get the ten framework/opcheck cases, covering active-gradient rejection, empty/offset inputs and non-default streams through their respective tests. Timing-path and bad-baseline rejection tests run under both comparison modes. CPU tests cover loader failure/idempotence and prevent mixing different comparisons in one statistical summary; CPU checks do not qualify CUDA execution.

Use [the geometry notebook](../../notebooks/05-t4-geometry128-experiment.ipynb). It runs both numerical/framework suites before collecting the required 72-case and safety 16-case matrices with `--comparison geometry128`. The output is `aegis-f04b-evidence.zip`.

The `f04-paired-v2` manifest, summaries and raw samples identify the comparison and exact arm operators. In geometry mode, the historical `reference` arm label means the retained 256-thread native operator; `native` means the 128-thread candidate. Both outputs are independently checked against the mathematical eager reference before timing. A numerical failure in either blocks measurement. The default eager comparison remains available.

The existing 100-call warmup, frozen repetitions, 30 paired trials, event/host boundaries and 10000 paired bootstrap resamples remain in force. Calibration diagnostics remain visible; the earlier run showed that a 10 ms target is not met by every later group. No observations are removed for falling below that target.

## Decision after evidence

Report every shape's candidate/baseline ratio and uncertainty, including regressions and inconclusive cases. No averaging away a losing workload. A passing candidate remains experimental until the evidence supports a separately reviewed dispatch decision. If it loses or provides no useful benefit, retain the 256-thread implementation. Profiler attribution, compiled/third-party comparisons, sanitizer qualification and independent reproduction remain separate gates; ordinary-operator timing does not establish pure-kernel or model speedups.
