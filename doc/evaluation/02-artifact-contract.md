# Evaluation artifact contract

Status: specified, schema version 1. [Example manifest](manifest.example.json) is a not-run template, not an execution record. Artifact validation is future test T-E01.

## Bundle layout

Each attempt exports `artifacts/<run_id>/` containing `manifest.json`, `samples.jsonl`, `correctness.json`, optional `quality.json`, `summary.json`, `environment.txt`, logs, and `checksums.sha256`. Full model weights, credentials and private prompts are excluded. Committed documentation holds summaries and approved small public evidence; large logs/traces are attached to a release or contributor artifact location with checksums.

run_id is UTC timestamp plus an eight-character random suffix. Never overwrite a previous run ID. A resumed run is a new attempt with `parent_run_id`; completed cases can be referenced, not silently appended to an already summarized run.

## Manifest

Required keys: schema_version, run_id, parent_run_id, status, started_at, finished_at, code_commit, protocol_version, contributor_id, provider, environment, model, configuration, artifacts, limitations. Valid status values: `not_run`, `running`, `completed`, `failed`, `interrupted`, `skipped`. `not_run` permits null runtime values and empty artifacts. Completed status requires real timestamps, commit, actual environment, all required cases and file checksums. A run may contain skipped optional baselines and still complete its required scope, but those skips must be enumerated.

environment contains OS, architecture, Python, torch, torch_cuda, nvcc, compiler, driver, GPU name/capability/UUID/memory, dependency-lock hash and build flags. Unavailable optional observations such as power clocks are null with a reason. GPU identity cannot be null in completed GPU runs. Model runs require model/tokenizer revision and attention/dtype; kernel-only runs use model=null.

configuration contains experiment, profile, backend modes, seeds, shapes or fixture hash, warmups, repetitions, trial count, context/output caps, timing boundaries, transport and concurrency where applicable. An ordinary operator benchmark and a preallocated diagnostic have distinct `timing_scope` values.

## Raw samples

Each JSONL record identifies run_id, case_id, trial_id, pair_id, arm, status and timing_scope. `arm` is `reference`, `native`, or a named baseline with an immutable version in the manifest. A completed kernel record contains rows, hidden, dtype, eps, repetitions, elapsed_ms and host_elapsed_ms; durations must be finite and nonnegative. Derived microseconds/call are `elapsed_ms * 1000 / repetitions` and must not replace the raw fields.

Model records include input_tokens, generated_tokens, prefill_ms, decode_step_ms array, host_total_ms, peak_allocated_bytes and peak_reserved_bytes. Serving records include request_id, arrival order, client monotonic timestamps relative to run start, HTTP status, outcome, prompt/generated counts, queue_ms and text-chunk timestamps. Absolute clocks from different machines are never subtracted. Failed/rejected requests retain outcome and available timing with missing values null, not zero.

Correctness records contain test/case ID, dtype/shape/seed, tolerance, error statistics, nonfinite mask results and pass/fail/skip reason. quality.json includes corpus hash, scored-token count, logit/NLL/KL statistics and diagnostic text/token differences. Store fixed public corpus outputs only.

## Summary and integrity

summary.json is derived from raw data and lists protocol version, case counts, required/optional failures, statistics, paired speedup intervals and claim status (`improved`, `regressed`, `inconclusive`, `not_measured`). A correctness failure forbids an `improved` release claim even if timing is lower.

Hash every payload file, including manifest, after finalizing it. The checksum index does not hash itself. Exclude the index from the manifest's payload list to avoid circular hashing. The manifest lists relative payload paths; the checksum index supplies their final hashes. On interruption, atomically write a terminal manifest for the exported snapshot. Never declare running files immutable or verified before finalization.

Redact tokens, authorization headers, personal filesystem paths and unrelated process details from exports. Retain only task-relevant GPU/process observations needed to assess interference.
