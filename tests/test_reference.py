"""Supplementary CPU semantics checks; GPU qualification lives in the notebook."""

import pytest
import torch

from aegis_norm import explain_dispatch, rms_norm


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_known_values_and_independent_output(dtype):
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=dtype)
    weight = torch.tensor([2.0, -1.0], dtype=dtype)
    x_before, weight_before = x.clone(), weight.clone()
    actual = rms_norm(x, weight, 1e-5)
    expected = torch.tensor([[6 / (12.5 + 1e-5) ** 0.5, -4 / (12.5 + 1e-5) ** 0.5], [0, 0]])
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-3)
    assert actual.dtype == dtype and actual.data_ptr() != x.data_ptr()
    torch.testing.assert_close(x, x_before)
    torch.testing.assert_close(weight, weight_before)


def test_fp16_cast_boundary_changes_result():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(3, 33, generator=generator).half()
    weight = torch.randn(33, generator=generator).half()
    normalized32 = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    expected = normalized32.half() * weight
    incorrectly_fused = (normalized32 * weight.float()).half()
    assert torch.any(expected != incorrectly_fused)
    torch.testing.assert_close(rms_norm(x, weight, 1e-5), expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(7,), (3, 33), (2, 3, 257), (0, 7)])
def test_shape_empty_and_odd_width(shape):
    x = torch.ones(shape)
    output = rms_norm(x, torch.ones(shape[-1]), 1e-5)
    assert output.shape == x.shape
    torch.testing.assert_close(output, torch.full_like(x, (1 + 1e-5) ** -0.5))


def test_noncontiguous_mixed_dtype_and_gradients():
    x = torch.ones(3, 4, requires_grad=True)
    sliced = x[:, ::2]
    weight = torch.ones(2, dtype=torch.float64, requires_grad=True)
    output = rms_norm(sliced, weight, 1e-5)
    assert output.dtype == torch.float64
    output.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(x.grad[:, 1::2]) == 0


@pytest.mark.parametrize("eps", [0, -1, float("nan"), float("inf"), 1e-50, 1e40, True, "1e-5"])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError, match="eps"):
        rms_norm(torch.ones(2), torch.ones(2), eps)


@pytest.mark.parametrize(
    "x,weight,error",
    [
        (torch.tensor(1.0), torch.ones(1), ValueError),
        (torch.ones(2, 0), torch.ones(0), ValueError),
        (torch.ones(2), torch.ones(3), ValueError),
        (torch.ones(2), torch.ones(1, 2), ValueError),
        (torch.ones(2, dtype=torch.int64), torch.ones(2), TypeError),
        (torch.ones(2), torch.ones(2, dtype=torch.complex64), TypeError),
        ([1, 2], torch.ones(2), TypeError),
        (torch.ones(2), torch.ones(2, device="meta"), ValueError),
    ],
)
def test_invalid_metadata(x, weight, error):
    with pytest.raises(error):
        rms_norm(x, weight, 1e-5)


def test_dispatch_is_honest_and_strict():
    args = (torch.ones(2), torch.ones(2), 1e-5)
    assert explain_dispatch(*args).reason == "native_rmsnorm_not_implemented"
    assert explain_dispatch(*args, backend="reference").reason == "explicit_reference"
    with pytest.raises(RuntimeError, match="not implemented"):
        rms_norm(*args, backend="cuda")
    with pytest.raises(ValueError, match="backend"):
        rms_norm(*args, backend="unknown")


def test_special_values():
    x = torch.tensor([[float("inf"), 1], [float("nan"), 1], [1e30, 1e30]])
    y = rms_norm(x, torch.ones(2), 1e-5)
    assert torch.isnan(y[0, 0]) and y[0, 1] == 0
    assert torch.isnan(y[1]).all()
    assert torch.equal(y[2], torch.zeros(2))
