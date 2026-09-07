"""Metadata implementation for the explicitly loaded native operator."""

import math
import struct

import torch


def fake_rms_norm(x, weight, eps):
    """Describe a fresh contiguous output without reading tensor values or GPU state."""
    torch._check(
        x.device.type == "cuda" and weight.device.type == "cuda",
        lambda: "x and weight must be CUDA tensors",
    )
    torch._check(x.device == weight.device, lambda: "x and weight must be on the same device")
    torch._check(
        x.layout == torch.strided and weight.layout == torch.strided,
        lambda: "x and weight must use strided layout",
    )
    torch._check(x.ndim >= 1, lambda: "x must have rank >= 1")
    width = x.shape[-1]
    torch._check(width > 0, lambda: "width must be positive")
    torch._check(width <= 65536, lambda: "width must be <= 65536")
    torch._check(weight.ndim == 1, lambda: "weight must have shape [H]")
    torch._check(weight.shape[0] == width, lambda: "weight must have shape [H]")
    torch._check(
        x.is_contiguous() and weight.is_contiguous(),
        lambda: "native RMSNorm requires contiguous inputs",
    )
    torch._check(
        x.dtype in (torch.float16, torch.float32), lambda: "native RMSNorm supports FP16 and FP32"
    )
    torch._check(x.dtype == weight.dtype, lambda: "x and weight must have the same dtype")
    torch._check(
        math.isfinite(eps) and 0 < eps <= torch.finfo(torch.float32).max,
        lambda: "eps must be positive finite FP32",
    )
    try:
        eps32 = struct.unpack("f", struct.pack("f", eps))[0]
    except (OverflowError, struct.error):
        eps32 = float("inf")
    torch._check(math.isfinite(eps32) and eps32 > 0, lambda: "eps must be positive finite FP32")
    torch._check(
        x.numel() // width <= 2**31 - 1, lambda: "row count exceeds the supported CUDA grid limit"
    )
    torch._check(
        not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad),
        lambda: "native RMSNorm is inference-only; use reference for active gradients",
    )
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def register_fake():
    """Called once by the loader, after the C++ schema exists."""
    torch.library.register_fake("aegis_norm::rms_norm", fake_rms_norm)
