# F03 contributor GPU evidence review

Status: initial framework registration qualification passed. Reviewed 2026-09-07. Tested commit: `ed512c97a084a6dc05c6d37502fbf435fc2802aa`; clean worktree. Run: `20260907T064135Z-a1b1bb4d`.

Preserved [original archive](runs/f03-20260907/evidence.zip) and [review manifest](runs/f03-20260907/review.json). All 17 top-level artifact hashes, five nested smoke hashes and 11 package source hashes verify against the archive and exact Git blobs. The collection reports no missing evidence. Fresh compiler commands for both extensions include sm_75 and link steps; registration tests then reuse that same build directory.

| Suite | Passed | Failures/errors/skips | Suite duration |
|---|---|---|---|
| Foundation | 38 | 0/0/0 | 101.845 s |
| Native RMSNorm | 223 | 0/0/0 | 44.382 s |
| Framework and metadata | 42 | 0/0/0 | 6.101 s |

These durations include test/build overhead and are not kernel benchmarks. The 42 framework cases comprise eight opcheck cases, two dynamic aot_eager cases and 32 metadata cases. Each opcheck case requires all four utilities to succeed. This exercises the changed C++ Autograd redispatch through the actual loaded operator while the numerical suite checks regression behavior.

Environment: Linux x86_64, Python 3.13.15, PyTorch 2.14.0+cu126, CUDA toolkit 12.6.85, GCC 11.4.0, Tesla T4 capability 7.5, driver 580.82.07. Preflight has no blockers; its `qualification: not_run` field describes preflight itself, before later execution. The standalone vector smoke passed separately and does not by itself qualify RMSNorm.

Dependency audits report no known vulnerabilities among audited packages. The installed audit cannot identify the unpublished project package or PyTorch's local cu126 label; the direct audit includes canonical torch 2.14.0. The missing NumPy warning did not prevent any required test; no NumPy conversion is used.

No runtime fix is indicated by this archive. F03 meets its initial T-K09 execution gate on this environment. Training, full-model compilation, sanitizer checks, independent reproduction and performance qualification remain separate. No speedup or production-readiness claim follows from these results.
