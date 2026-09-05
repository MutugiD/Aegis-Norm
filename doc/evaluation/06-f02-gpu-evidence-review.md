# F02 T4 build and correctness review

Reviewed 2026-09-06. Contributor execution: 2026-09-05 UTC. **223 native RMSNorm tests and 38 foundation tests passed**, with zero failures, errors or skips in both JUnit reports.

## Evidence and verification

The [original archive](runs/f02-20260905/evidence.zip) is preserved byte-for-byte. [Review metadata](runs/f02-20260905/review.json) records its SHA-256 and verification counts. The tested commit is `d8f5ee5c129c66e7db5214a1cf4522df58eb525f`; its worktree report is clean.

All 15 top-level artifact hashes and five nested smoke-artifact hashes match the archive contents. All ten recorded Python/native source hashes match the files in the tested Git commit. The collection reports no missing files. These integrity checks establish consistency with the supplied files, not independent execution attestation.

Unlike the partial F01 archive, this archive includes preflight, a resolved environment version inventory, smoke status/check records, native compiler/test logs, JUnit results and hashes. Compiler logs show separate C++ compilation, nvcc compilation for `sm_75`, and linking for both native extensions; this is evidence of compilation rather than just loading a cached binary.

## Observed environment

| Component | Reported value |
|---|---|
| Device | Tesla T4, capability 7.5 |
| OS / Python | Linux x86_64 / 3.13.15 |
| PyTorch | 2.14.0+cu126 |
| Driver | 580.82.07; driver display CUDA maximum 13.0 |
| Toolkit compiler | nvcc 12.6.85 |
| Host compiler | GCC 11.4.0 |
| Build flags | C++20, O2, sm_75, no global fast-math flag |

Only this recorded environment is exercised by this run. The broader candidate Python range is not automatically qualified.

## Results and interpretation

The native suite passes the implemented FP16/FP32 width/rank cases, distributions and epsilons, exceptional values, empty/offset inputs, stream ordering, mutation/output checks, invalid arguments and gradient behavior. Tolerances remain those specified before execution. No failed numerical cases require a kernel correction based on this archive.

JUnit durations are 42.421 seconds for the native suite and 94.615 seconds for the foundation suite. They include test/fixture overhead and are **not kernel latency or speedup measurements**. Per-element error distributions are not exported by this initial pytest suite.

Both dependency audits report zero known vulnerabilities among inspected packages. The installed audit skips the unpublished project and suffixed torch wheel; the direct audit checks torch 2.14.0 separately. The NumPy availability warning does not fail the tests and does not affect this tensor-only suite.

## Remaining scope

This result supports integrating the initial native forward implementation. It does not establish complete release readiness. T-K08 sanitizer evidence, extended lifetime/profiling and huge-grid coverage, independent reproduction, a portable qualified dependency lock, model integration and performance evaluations remain outstanding. The resolved environment inventory is retained as evidence and is not relabeled as a portable hash lock.

The older F01 partial-archive review remains historical; this subsequent run supplies the missing standalone artifacts for the new tested commit. F03 framework registration checks are the next increment. No native-kernel change was needed during this evidence review.
