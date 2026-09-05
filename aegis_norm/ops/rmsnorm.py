"""Cast-ordered reference and explicit dispatch to the native RMSNorm operator."""

import math
import numbers
import struct
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DispatchDecision:
    backend: str
    reason: str


def _validate(x, weight, eps, backend):
    if backend not in ("auto", "cuda", "reference"):
        raise ValueError("backend must be auto, cuda, or reference")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be torch.Tensor instances")
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError("x and weight must use strided layout")
    if not x.is_floating_point() or not weight.is_floating_point():
        raise TypeError("x and weight must have floating-point dtypes")
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("x must have rank >= 1 and a nonempty final dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must have shape [x.shape[-1]]")
    if x.device != weight.device:
        raise ValueError("x and weight must be on the same device")
    if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
        raise ValueError("eps must be a positive finite real number")
    try:
        eps32 = struct.unpack("f", struct.pack("f", float(eps)))[0]
    except (OverflowError, struct.error):
        raise ValueError("eps must be representable as positive finite FP32") from None
    if not math.isfinite(eps32) or eps32 <= 0:
        raise ValueError("eps must be representable as positive finite FP32")
    return float(eps)


def explain_dispatch(x, weight, eps, *, backend="auto") -> DispatchDecision:
    """Validate inputs and expose the execution choice without running tensor math."""
    _validate(x, weight, eps, backend)
    if backend == "reference":
        return DispatchDecision("reference", "explicit_reference")
    reason = _native_reason(x, weight)
    if reason:
        if backend == "cuda":
            raise RuntimeError(f"Native RMSNorm unavailable: {reason}; call load_native on T4")
        return DispatchDecision("reference", reason)
    return DispatchDecision("cuda", "native_rmsnorm")


def _native_reason(x, weight):
    from ..native import is_loaded

    if x.device.type != "cuda":
        return "unsupported_device"
    if x.dtype not in (torch.float16, torch.float32) or x.dtype != weight.dtype:
        return "unsupported_dtype"
    if not x.is_contiguous() or not weight.is_contiguous():
        return "noncontiguous_input"
    if x.shape[-1] > 65536:
        return "unsupported_width"
    if x.numel() // x.shape[-1] > 2**31 - 1:
        return "unsupported_row_count"
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return "active_gradients"
    if not is_loaded():
        return "native_extension_not_loaded"
    if torch.cuda.get_device_capability(x.device) != (7, 5):
        return "unsupported_capability"
    return None


def rms_norm(x, weight, eps, *, backend="auto") -> torch.Tensor:
    """Normalize over the final axis, preserving the Llama cast-before-weight rule.

    Reference tensor operations execute on x.device, including CUDA when x is
    on the T4. Normal PyTorch autograd and weight dtype promotion are retained.
    Native mode requires prior explicit loading. This function never compiles code.
    """
    decision = explain_dispatch(x, weight, eps, backend=backend)
    if decision.backend == "cuda":
        return torch.ops.aegis_norm.rms_norm(x, weight, float(eps))
    x32 = x.float()
    inverse_rms = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + float(eps))
    normalized = (x32 * inverse_rms).to(x.dtype)
    return weight * normalized
