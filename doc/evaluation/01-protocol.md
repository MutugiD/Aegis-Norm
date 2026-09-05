# Evaluation protocol: kernel, model and serving

Status: specified; all performance results are **not yet measured**. Run correctness first. Experiments use immutable code/model identifiers and the same environment/configuration in both arms. This protocol distinguishes API overhead from device work and text chunks from model tokens.

## Run discipline

Use seeds 0, 17 and 123 for correctness; use benchmark ordering seed 2026. Record GPU UUID/name/capability, driver, temperature, clock/power information where available, competing processes, Python, torch/CUDA/nvcc/compiler, package lock, build flags, repository commit, model/tokenizer revision, attention backend, dispatch mode, workload and protocol version. Do not assume a free GPU is exclusive or fixed-clock.

Each run has a unique ID and artifact manifest. Finish and export each case before advancing. A disconnected session leaves completed cases valid but the run incomplete. Do not fill missing observations. Compare contributors within matched environments and report differences across sessions rather than pooling all samples into one distribution.

Use strict native mode and confirm the expected norm sites were adapted. Disable diagnostic counters during timing after a separate dispatch verification. Reference and candidate use the same dtype, epsilon, model weights, attention implementation, generation limits and input data. Compilation/download/warmup time is reported separately, never hidden inside steady-state latency.

## Kernel evaluation: EXP-KERNEL

Required baselines: pinned cast-ordered eager expression and attempted native `torch.nn.functional.rms_norm` with explicit eps. Verify semantics before comparing; where casting differs, report it as a related implementation, not the identical numerical baseline. Attempt compiled eager and at least one existing fused candidate when the environment supports them; otherwise export exact unavailable reasons and versions. Never count fallback as native.

| Profile | Shapes and use |
|---|---|
| Smoke | Rows 1, 32, 128; H 2048, 4096; FP16; build/correctness only, no headline speedup |
| Required | Rows 1, 2, 4, 8, 16, 32, 128, 512, 2048; H 2048, 4096, 5120, 8192; FP16/FP32 |
| Scalar safety | H 96, 257, 1023, 4097; rows 1 and 128; offset/unaligned inputs; report separately |

Generate identical seeded tensors per case and reuse them for paired trials. Warm up each arm 100 calls. Calibrate repetitions with a short warmup so a timed group targets at least 10 ms; clamp repetitions to 20..10000 and freeze the same count for both arms in that case. Collect 30 paired trials with seeded AB/BA order to limit drift. Export total event duration and repetitions, not just the divided latency.

Record CUDA events on the current stream around the repeated operator calls and synchronize the end event before reading elapsed time. This measures elapsed device-stream time for the operator sequence; Python submission gaps can appear in it. Also time the synchronized host operator sequence with a monotonic clock. Both normal operator arms allocate outputs. Use a separate explicitly labeled preallocated/native-kernel experiment or profiler to isolate pure kernel duration; do not call event timing a guaranteed hardware-only measurement.

Default case reuses inputs and gamma and is labeled warm-cache. An optional rotating-buffer case is reported separately. Effective bytes/sec uses a declared logical-byte model and must not be presented as measured DRAM bandwidth. Actual traffic, register count/spills, occupancy and launch count come from separate diagnostic profiling when available. Profiled durations are excluded from main timing tables.

## Model evaluation: EXP-MODEL

Use pinned TinyLlama, FP16, eager attention, one GPU, no offload/quantization, eval plus inference mode. Compare the same loaded weights sequentially using patch/restore; clear request caches between trials and warm each mode separately. Loading and model duplication are excluded from steady-state comparisons but reported in setup/memory notes.

Performance prompt lengths: 16, 128, 512 and 1536 input tokens. Build exact-length valid token sequences by repeating the tokenized neutral paragraph `A GPU executes many threads that operate on data in memory.` and slicing, retaining the model's normal required initial token. Save token IDs and their hash. These synthetic inputs measure shape effects, not response quality. Use 64 generated tokens for required latency/TPS comparison, with EOS disabled only in this synthetic performance mode and explicitly reported. All runs fit the 2048-token context.

Warm up 3 complete runs per shape/mode, then collect 20 paired runs. Separate first forward/prefill device time, first token selection, and the following 63 cached decode steps. First new token comes from prefill logits. Record host completion time, per-step intervals, generated count and direct model throughput. Never count prompt tokens as generated tokens. Peak allocated/reserved memory is reset after model load/warmup and measured over a fresh request, with persistent model memory also reported separately. Collect device-free-memory observations when available.

Profile one separate representative request to estimate RMSNorm runtime share, verify operator count and distinguish CPU/GPU overhead. Use the measured share in an Amdahl calculation, clearly labeled as an optimistic attribution estimate. Never add overlapping asynchronous profiler spans as if they were serial wall time.

## Model numerical and output evaluation

Use [fixtures](fixtures.json), version 1, as a small regression corpus, not a general model-quality benchmark. For teacher forcing, tokenize each `prompt + "\n" + continuation` as plain text and score every next token after the first. Use the same full token sequence in both arms. In a separate generation check, put prompt into the chat template and generate up to 64 tokens greedily with normal EOS.

Compare finite logits over every scored position, shifted next-token NLL and reference-to-candidate KL computed in FP32. Initial gates: maximum logit absolute difference <= 0.1, RMS logit difference <= 0.02, absolute mean NLL difference <= 0.02 nats/token, mean KL <= 1e-3. These are predeclared project regression tolerances, not validated quality thresholds. Kernel numerical gates must also pass. Failure blocks qualification pending investigation; changes to tolerances require a reviewed protocol revision, not silent relaxation.

Record greedy token agreement, first divergence position, reference top-two logit margin at divergence, generated lengths and both texts for this public corpus. Identical text is not guaranteed by close floating-point results; text divergence alone is diagnostic if numeric gates pass. Do not claim unchanged model quality from this small corpus. Broader task accuracy/perplexity evaluations are a later model-qualification feature.

## Serving evaluation: EXP-SERVE

Compare two separately configured server runs, reference and native, with identical weights/protocol/queue settings. Start and warm each before client timing. Required transport is localhost inside the GPU session; optional tunnel results have their own run IDs and cannot be substituted for local model evidence.

Use fixture prompts, normal EOS, max_tokens 64, temperature 0. For concurrency 1 and 5, run three rounds of 32 completed-or-terminal attempts per mode; closed-loop clients submit the next request when the previous one terminates. Record attempt IDs/order and include failures/rejections. Concurrency 6 is an overload functionality scenario with a fake held worker; real runs may complete quickly enough that every arrival is not rejected. Do not promise a particular real-model rejection count.

Record dispatch, headers, first nonempty content, each content chunk, terminal outcome, completion time, token counts from server artifacts, queue/active time and HTTP status. Client TTFT includes transport and queueing. SSE chunks are not tokens; report chunk intervals separately. Aggregate TPS is successful generated token count divided by the whole workload wall interval. Also report success ratio, rejected ratio, request completion p50/p95 and per-request lengths. p95 on a small sample is descriptive; do not claim production p99 or SLA evidence.

Server token ITL comes from instrumented generation events, not chunk timestamps. Instrumentation overhead is measured in a separate control run; do not assume it is free. No operational cost savings claim follows from free GPU timing alone.

## Statistics and reporting

For each case report raw sample count, median, p95, min/max and interquartile range. Define speedup as reference time / candidate time. Bootstrap paired trial indices 10000 times with seed 2026 to produce a 95% interval for the ratio of paired-arm median times. Preserve pairing and do not bootstrap individual inner repetitions as independent observations.

Call a latency improvement supported only when the interval lies entirely above 1 and correctness passes. An interval overlapping 1 is inconclusive; below 1 indicates regression. No post-hoc outlier removal. Exclude only invalid/corrupted runs with recorded reasons, retaining their artifacts. Report clock/temperature drift, OOM and competing process observations as limitations. Avoid a single average across shapes that conceals losses.

First-release evidence requires one complete qualifying T4 run plus an independent contributor reproduction of setup/correctness and the key claimed workload results. Lack of a second contributor prevents the label independently reproduced, not publication of clearly labeled preliminary observations.
