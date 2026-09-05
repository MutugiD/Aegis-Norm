# Normalization and integration contracts

Status: specified for v0.1; not an implemented API. [Kernel PRD](../prd/01-kernel-library.md) owns requirements.

## Public API

```python
rms_norm(x, weight, eps, *, backend="auto") -> Tensor
patch_model(model, *, backend="auto") -> PatchHandle
handle.restore() -> None
```

`backend` is exactly `auto`, `cuda`, or `reference`; invalid values raise `ValueError`. The raw registered schema is `aegis_norm::rms_norm(Tensor x, Tensor weight, float eps) -> Tensor`, callable through `torch.ops.aegis_norm.rms_norm`. It is functional: no mutation, no aliasing, one output.

## Numerical semantics

For each row, cast x to FP32; calculate mean of squared values; multiply x by `rsqrt(mean + eps)` in FP32; round the normalized values to x dtype; multiply by weight; return x dtype on the native same-dtype path. This preserves the selected Llama casting boundary. Do not replace it with one FP32 multiplication by gamma followed by a final half cast without a separate semantic decision. [Pinned reference](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py).

Reference mode uses that expression and PyTorch's normal promotion rules when weight dtype differs. FP32 summation order may differ in native execution; the testing protocol defines tolerances. No bitwise equivalence or token identity promise is made.

## Inputs and dispatch

| Condition | Native operator | Public auto / integration |
|---|---|---|
| x rank >= 1, final width 1..65536 | Supported | Native if other checks pass |
| Contiguous strided x; contiguous weight `[H]` | Required | Noncontiguous tensors use reference without hidden `.contiguous()` copy |
| FP16 or FP32, x/weight same dtype | Supported | Other floating or mixed dtypes use reference |
| Same CUDA device, capability 7.5 | Required for v0.1 qualified path | CPU/other devices use reference until separately qualified |
| Positive finite eps, representable as positive finite FP32 | Required | Invalid epsilon raises `ValueError` before dispatch |
| No active gradient requirement | Required | When grad is enabled and either tensor requires grad, use reference |
| Zero leading dimension, H > 0 | Return correctly shaped empty tensor, no launch | Same |
| Storage offset causes misalignment | Scalar-safe native path | No copy solely for alignment |
| Shape mismatch, nonfloating inputs, incompatible devices | Error | `ValueError` for metadata; `TypeError` for dtype category |
| Extension unavailable / unsupported valid native case | Error in `cuda` mode | Reference with inspectable reason in `auto` mode |

For raw `torch.ops` calls, invalid metadata or active gradient requirements raise `RuntimeError` with a specific condition; no backward kernel is registered. FakeTensor support returns matching output metadata and validates shape/dtype relationships without reading values. Test registration with opcheck under inference conditions; numerical tests remain separate.

Use 64-bit host/index arithmetic. Reject row count above the supported grid limit (2^31-1) before launch. Output preserves shape, dtype and device but is contiguous; it need not preserve a nonstandard stride because native input is contiguous. Invalid shape is not silently repaired by fallback. Native launches never catch a CUDA fault and retry on a damaged context.

NaN/Inf inputs are not scanned on the host. Native special-value propagation must match the reference classifications; large FP32 inputs can overflow the reference's sum of squares. These are tested separately from finite normal-range tolerance checks.

## Initial kernel and launch

Assign one block per row and 256 threads per block. Each thread accumulates a strided sequence in FP32. All 32 lanes participate in each warp reduction; unused work contributes zero. Lane zero writes each of eight warp sums to shared memory. After a block barrier, the first warp reads those eight values and zero-fills remaining lanes, then executes offsets 16, 8, 4, 2, 1. Thread zero writes the inverse RMS; a second block barrier precedes scaling.

Use matching `scalar_t` pointers, FP32 conversion before squaring, a device guard, the current CUDA stream for that device, and a launch-error check without device-wide synchronization. Synchronization belongs to tests/timing or explicit runtime dependencies. Output allocation uses framework allocation. No `cudaMalloc` per row and no input mutation.

Do not enable global `--use_fast_math` by default. Packed loads require validation of every relevant pointer, row stride and tail, not merely base alignment. Compare optimized variants against this implementation and the reference before dispatching them by shape.

## Reversible Llama integration

Only exact supported `LlamaRMSNorm` instances in the pinned Transformers version are adapted; preserve the module object, weight Parameter identity, epsilon, state-dict keys, dtype and device. Bind an instance-local forward method and retain its original bound method in the handle. Never replace a Transformers class globally.

Validate candidate modules before mutation. If strict `cuda` mode finds an unsupported norm, fail transactionally and restore any touched instances. Repeated `patch_model` with the same backend returns the existing live handle; a different backend requires restore first. Restoring is idempotent. Reject conflicting third-party instance patches; do not overwrite them. A model with no supported modules returns a report with zero patched modules in auto mode and raises in strict mode.

The report exposes patched paths/count and fallback reasons, not prompt contents. Static unsupported cases are reported once; optional per-call counters are disabled during latency measurements. Training or active autograd executes the retained original forward in auto mode, while strict native mode rejects it. Test `model.train()` and a trainable input explicitly.

`save_pretrained` must preserve ordinary parameter state. A separately loaded model starts unpatched, then can be adapted again. Full Python object pickling, export, arbitrary model families and full-model compile are not v0.1 guarantees. Patch before any optional compile experiment and restore only after discarding that experiment's compiled callable.
