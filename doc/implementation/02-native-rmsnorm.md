# F02: native RMSNorm forward

Status: implementation ready for T4 qualification; native RMSNorm compilation, numerical results and sanitizer runs are not yet executed. F01's passing vector smoke does not validate this kernel.

## Explicit load and dispatch

[load_native](../../aegis_norm/native.py) explicitly compiles the two RMSNorm source files and loads their dispatcher registration in the current Python process. It is idempotent after success and refuses a conflicting existing registration. Importing the package or calling the reference does not compile code. Source changes after loading require a new process.

```python
import torch
from aegis_norm import load_native, rms_norm, explain_dispatch

load_native()
with torch.inference_mode():
    x = torch.randn(3, 2048, device="cuda", dtype=torch.float16)
    weight = torch.ones(2048, device="cuda", dtype=torch.float16)
    decision = explain_dispatch(x, weight, 1e-5)
    output = rms_norm(x, weight, 1e-5, backend="cuda")
```

The registered schema is `aegis_norm::rms_norm(Tensor x, Tensor weight, float eps) -> Tensor`. The Python wrapper validates metadata before dispatch. Auto mode exposes reference fallback reasons for unsupported device, dtype, layout, width, row count, active gradients or an unloaded extension. Strict CUDA mode raises instead. A native launch failure propagates; it is never caught and retried as reference execution.

## C++ boundary

The [binding](../../aegis_norm/csrc/rmsnorm_binding.cpp) validates both tensors on one CUDA device, contiguous strided storage, matching FP16/FP32 dtype, weight shape `[H]`, width 1..65536, valid FP32 epsilon, inference conditions and the grid row limit. The initial architecture target is capability 7.5. The same validation protects raw `torch.ops` calls. CUDA and Autograd dispatch registrations use the checked forward function; active gradients are explicitly rejected. Reference mode retains ordinary PyTorch autograd.

This is dispatcher registration, not full compiler integration. FakeTensor/opcheck support remains F03; full-model compilation and a backward kernel are not promised.

## Reduction and memory behavior

The [kernel](../../aegis_norm/csrc/rmsnorm_cuda.cu) assigns one 256-thread block per row. Each thread reads a strided portion and accumulates squared values in FP32. All 32 lanes participate in each warp shuffle. Eight warp sums are written to shared memory, followed by a barrier. The first warp combines those eight values with zero-filled remaining lanes and stores the inverse RMS. A second barrier makes it visible before output scaling.

The input is read again during scaling. This eliminates intermediate global-memory tensors but does not imply exactly one DRAM read; caches affect actual traffic. FP16 normalization rounds to half before multiplication by gamma, matching the selected reference casting boundary. Scalar indexing supports odd widths and contiguous storage offsets without alignment copies. Indices use 64-bit arithmetic; empty leading dimensions return an allocated empty output without a kernel launch.

The launcher uses a device guard, PyTorch output allocation, the device's current CUDA stream and a launch-error check. It does not synchronize the GPU. Tests establish completion before examining values. No global fast-math flag or packed-load optimization is enabled.

## Notebook and acceptance tests

Run [the F02 notebook](../../notebooks/02-t4-rmsnorm-correctness.ipynb) at a recorded commit. It performs the repaired F01 setup and metadata collection, then runs [native correctness tests](../../tests/test_native_rmsnorm.py) with compiler output retained in `native-build-and-tests.log` and JUnit in `rmsnorm-tests.xml`. The final archive includes an artifact hash inventory and reports missing steps.

Coverage includes the specified width matrix, row/rank cases, three seeds and epsilons, multiple distributions, FP16/FP32, exceptional values, empty/offset inputs, input preservation, output ownership, side streams, invalid raw arguments and active-gradient behavior. Acceptance uses the existing FP32 `(atol=1e-6, rtol=2e-5)` and FP16 `(2e-3, 2e-3)` tolerances; exceptional-value classifications are compared separately.

This is initial T-K01..T-K07 coverage. It does not establish all release gates: huge-grid metadata testing, multi-device execution, extended lifetime/profiling checks and T-K08 sanitizer evidence remain outstanding. On a host with compute-sanitizer, invoke the installed notebook environment's Python under memcheck, racecheck and synccheck, selecting representative native tests. Unavailable tooling is pending rather than a pass. Do not report benchmark speedups from suite duration.

## Sources

The implementation follows the project [operator contract](../architecture/02-operator-contract.md) and PyTorch's [custom operator registration guidance](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), [dispatcher guidance](https://docs.pytorch.org/tutorials/advanced/dispatcher) and [explicit extension loader](https://docs.pytorch.org/docs/main/cpp_extension.html), checked 2026-09-06. GPU results remain pending execution.
