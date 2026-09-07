"""Metadata checks use fake CUDA tensors; no GPU allocation or compilation."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from aegis_norm.registration import fake_rms_norm


@pytest.fixture(scope="module")
def fake_op():
    # An isolated schema tests registration without impersonating the native loader.
    library = torch.library.Library("aegis_metadata_test", "DEF")
    try:
        library.define("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor")
        torch.library.register_fake("aegis_metadata_test::rms_norm", fake_rms_norm, lib=library)

        def autograd(x, weight, eps):
            if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
                raise RuntimeError("native RMSNorm is inference-only")
            with torch._C._AutoDispatchBelowAutograd():
                return torch.ops.aegis_metadata_test.rms_norm(x, weight, eps)

        library.impl("rms_norm", autograd, "Autograd")
        yield torch.ops.aegis_metadata_test.rms_norm.default
    finally:
        library._destroy()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(7,), (2, 257), (2, 3, 32), (0, 7), (2**31 - 1, 1)])
def test_fake_metadata_and_fresh_storage(fake_op, dtype, shape):
    with FakeTensorMode():
        x = torch.empty(shape, device="cuda", dtype=dtype)
        weight = torch.empty(shape[-1], device="cuda", dtype=dtype)
        y = fake_op(x, weight, 1e-5)
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
        assert y.is_contiguous() and y.storage_offset() == 0
        assert not torch._C._is_alias_of(y, x)
        assert not torch._C._is_alias_of(y, weight)


def test_fake_offset_is_not_preserved(fake_op):
    with FakeTensorMode() as mode:
        x = FakeTensor(mode, torch.empty(15, device="meta")[1:].view(2, 7), torch.device("cuda:0"))
        y = fake_op(x, torch.empty(7, device="cuda"), 1e-5)
        assert x.storage_offset() == 1 and y.storage_offset() == 0


@pytest.mark.parametrize(
    "case, message",
    [
        ("rank", "rank"),
        ("zero_width", "width"),
        ("large_width", "width"),
        ("weight_rank", "shape"),
        ("weight_size", "shape"),
        ("stride", "contiguous"),
        ("weight_stride", "contiguous"),
        ("dtype", "FP16 and FP32"),
        ("mixed_dtype", "same dtype"),
        ("cpu", "CUDA"),
        ("device", "same device"),
        ("grid", "grid limit"),
        ("grad", "inference-only"),
    ],
)
def test_fake_rejects_invalid_metadata(fake_op, case, message):
    with FakeTensorMode():
        x, weight = torch.empty(2, 7, device="cuda"), torch.empty(7, device="cuda")
        if case == "rank":
            x = torch.empty((), device="cuda")
        elif case == "zero_width":
            x = torch.empty(2, 0, device="cuda")
        elif case == "large_width":
            x = torch.empty(2, 65537, device="cuda")
        elif case == "weight_rank":
            weight = torch.empty(1, 7, device="cuda")
        elif case == "weight_size":
            weight = torch.empty(6, device="cuda")
        elif case == "stride":
            x = torch.empty_strided((2, 7), (1, 2), device="cuda")
        elif case == "weight_stride":
            weight = torch.empty_strided((7,), (2,), device="cuda")
        elif case == "dtype":
            x = x.to(torch.bfloat16)
        elif case == "mixed_dtype":
            weight = weight.half()
        elif case == "cpu":
            x = torch.empty(2, 7)
        elif case == "device":
            weight = torch.empty(7, device="cuda:1")
        elif case == "grid":
            x, weight = torch.empty(2**31, 1, device="cuda"), torch.empty(1, device="cuda")
        elif case == "grad":
            x.requires_grad_(True)
        with pytest.raises(RuntimeError, match=message):
            fake_op(x, weight, 1e-5)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf"), 1e40, 1e-50, 3.4028235e38])
def test_fake_rejects_epsilon(fake_op, eps):
    with FakeTensorMode():
        with pytest.raises(RuntimeError, match="FP32"):
            fake_op(torch.empty(7, device="cuda"), torch.empty(7, device="cuda"), eps)


def test_symbolic_shapes_remain_symbolic(fake_op):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    mode = FakeTensorMode(shape_env=ShapeEnv())
    symbolic = mode.from_tensor(torch.empty(3, 7), static_shapes=False)
    with mode, torch.no_grad():
        x = torch.empty(symbolic.shape, device="cuda")
        weight = torch.empty(symbolic.shape[-1], device="cuda")
        y = fake_op(x, weight, 1e-5)
        assert all(isinstance(dim, torch.SymInt) for dim in y.shape)
        assert y.shape == x.shape
