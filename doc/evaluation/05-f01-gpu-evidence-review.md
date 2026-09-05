# F01 GPU smoke evidence review

Reviewed 2026-09-06; contributor execution dated 2026-09-05. Result: **38 tests passed, including both GPU smoke tests**. Full F01 artifact qualification remains incomplete.

## Received evidence and provenance

- [Original evidence archive](runs/f01-20260905/evidence.zip): retained byte-for-byte.
- [Executed contributor notebook](runs/f01-20260905/executed-notebook.ipynb): retained with its actual cells and outputs, separately from the maintained notebook.
- [Review metadata and received file hashes](runs/f01-20260905/review.json).

The archive identifies commit `a50b8ce6d88fb9c8419490a9fe984f2e2cf7cabf`. The notebook includes interactive setup repairs, so it is not an unmodified run of that commit's notebook. The package/native sources and the three executed test files match the subsequent setup-fix revision `d43341c`; this does not make the modified notebook a fresh reproduction of that revision.

## Observed results

JUnit reports 38 tests, zero failures, zero errors and zero skips, in 97.574 seconds. The console log agrees. This elapsed time is the complete test-suite duration, not kernel latency or a measured compilation duration.

The passing GPU cases are `test_native_execution_and_cuda_reference` and `test_native_binding_rejects_invalid_arguments`. Their fixture invokes the native builder/loader and the checks execute the vector smoke extension, test its side-stream behavior and run FP16/FP32 reference RMSNorm on CUDA tensors. The second case rejects invalid binding arguments. The other 36 tests check reference/preflight behavior and do not all execute on the GPU.

The logs identify Python 3.13.15, PyTorch 2.14.0+cu126, nvcc 12.6.85 and GCC 11.4.0; device output identifies a Tesla T4. Package installation and `pip check` succeed. Both audit JSON files have empty vulnerability lists for inspected packages. The installed audit skips the unpublished project package and the suffixed torch wheel; the direct-pin audit separately checks torch 2.14.0. This is dependency-advisory coverage, not proof of security for every native binary.

The NumPy-not-installed warning did not fail these tests. No NumPy conversion is required by this smoke suite.

## Missing evidence and cause

The archive contains only `setup-and-tests.log`, `commit.txt`, `installed-audit.json`, `direct-audit.json` and `tests.xml`. The executed notebook's preflight/build cell was replaced by dependency installation during interactive repair. Neither the standalone preflight CLI nor the standalone smoke CLI appears in its command log.

Consequently, `preflight.json`, the standalone smoke result/check records, compiler log, resolved environment snapshot and run hashes are absent. Pytest captured the successful fixture's compiler output instead of exporting it. Native execution passed, but this archive cannot independently establish a fresh compilation rather than a cache load or complete the environment/reproduction gate. Do not reconstruct those missing files as if the contributor had produced them.

## Follow-up

The maintained notebook keeps setup and preflight/build as separate cells. Its export cell now lists missing evidence instead of silently producing an apparently complete archive.

In the still-active contributor session, run the original preflight/build cell and export again:

```python
run([python, "-m", "aegis_norm.preflight", "--output",
     str(artifacts / "preflight.json"), "--require-t4"], cwd=repo)
run([python, "-m", "aegis_norm.smoke", "--output-root", str(artifacts)], cwd=repo)
```

This supplemental run may reuse the compiled extension cache; label it accordingly. A fresh-session build log is still needed for the fresh-compilation claim. If the session has ended, run the maintained notebook from a recorded revision and return the new archive. Native RMSNorm, model integration, benchmarks and release qualification remain pending; the passing vector smoke is not an RMSNorm performance result.
