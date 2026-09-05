"""F02 numerical/contract checks; opt in inside a T4 notebook."""

import os

import pytest
import torch

from aegis_norm import explain_dispatch, load_native, rms_norm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("AEGIS_RUN_GPU") != "1", reason="T4 notebook opt-in required"
    ),
]

WIDTHS = [1, 7, 31, 32, 33, 96, 160, 255, 256, 257, 768, 1023, 1024, 2048, 4096, 5120, 8192, 65536]


@pytest.fixture(scope="module", autouse=True)
def native_operator():
    load_native()


def compare(x, weight, eps=1e-5):
    before_x, before_weight = x.clone(), weight.clone()
    actual = rms_norm(x, weight, eps, backend="cuda")
    expected = rms_norm(x, weight, eps, backend="reference")
    atol, rtol = (2e-3, 2e-3) if x.dtype == torch.float16 else (1e-6, 2e-5)
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(actual[finite], expected[finite], atol=atol, rtol=rtol)
    torch.testing.assert_close(x, before_x, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(weight, before_weight, atol=0, rtol=0, equal_nan=True)
    assert actual.shape == x.shape and actual.dtype == x.dtype and actual.device == x.device
    assert actual.is_contiguous()
    if x.numel():
        assert actual.data_ptr() != x.data_ptr() and actual.data_ptr() != weight.data_ptr()
    return actual


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("rows,rank", [(1, 1), (3, 2), (32, 3)])
def test_widths_rows_and_ranks(dtype, width, rows, rank):
    generator = torch.Generator(device="cuda").manual_seed(17)
    shape = (width,) if rank == 1 else ((rows, width) if rank == 2 else (1, rows, width))
    x = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    weight = torch.rand(width, device="cuda", dtype=dtype, generator=generator) * 4 - 2
    with torch.inference_mode():
        compare(x, weight)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("seed", [0, 17, 123])
@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-3])
@pytest.mark.parametrize(
    "distribution", ["normal", "uniform", "constant", "alternating", "small", "large"]
)
def test_distributions_and_epsilon(dtype, seed, eps, distribution):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(3, 257, device="cuda", dtype=dtype, generator=generator)
    if distribution == "uniform":
        x = torch.rand(3, 257, device="cuda", dtype=dtype, generator=generator) * 20 - 10
    elif distribution == "constant":
        x.fill_(3)
    elif distribution == "alternating":
        x.fill_(7)
        x[:, ::2] *= -1
    elif distribution == "small":
        x *= 1e-3
    elif distribution == "large":
        x *= 1e3
    weight = torch.linspace(-2, 2, 257, device="cuda", dtype=dtype)
    with torch.inference_mode():
        compare(x, weight, eps)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_special_values(dtype):
    with torch.inference_mode():
        for value in (
            0,
            torch.finfo(dtype).tiny / 2,
            torch.finfo(dtype).max,
            float("inf"),
            -float("inf"),
            float("nan"),
        ):
            x = torch.full((3, 33), value, device="cuda", dtype=dtype)
            compare(x, torch.ones(33, device="cuda", dtype=dtype))
        x = torch.ones(3, 33, device="cuda", dtype=dtype)
        for value in (0, float("inf"), -float("inf"), float("nan")):
            compare(x, torch.full((33,), value, device="cuda", dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_empty_and_unaligned_contiguous_inputs(dtype):
    with torch.inference_mode():
        compare(
            torch.empty(0, 7, device="cuda", dtype=dtype), torch.ones(7, device="cuda", dtype=dtype)
        )
        x = torch.arange(3 * 33 + 1, device="cuda", dtype=dtype)[1:].reshape(3, 33)
        weight = torch.linspace(-1, 1, 34, device="cuda", dtype=dtype)[1:]
        compare(x, weight)


def test_side_stream_ordering():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.inference_mode(), torch.cuda.stream(stream):
        x = torch.empty(3, 257, device="cuda").fill_(3)
        weight = torch.ones(257, device="cuda")
        actual = rms_norm(x, weight, 1e-5, backend="cuda") + 2
        expected = rms_norm(x, weight, 1e-5, backend="reference") + 2
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)


def test_auto_and_raw_validation():
    x = torch.ones(3, 33, device="cuda")
    weight = torch.ones(33, device="cuda")
    raw = torch.ops.aegis_norm.rms_norm
    with torch.inference_mode():
        assert explain_dispatch(x, weight, 1e-5).backend == "cuda"
        sliced = x[:, ::2]
        assert explain_dispatch(sliced, weight[::2], 1e-5).reason == "noncontiguous_input"
        torch.testing.assert_close(
            rms_norm(sliced, weight[::2], 1e-5),
            rms_norm(sliced, weight[::2], 1e-5, backend="reference"),
        )
        invalid = [
            (x, weight.half(), 1e-5),
            (x[:, ::2], weight[::2], 1e-5),
            (x, weight.cpu(), 1e-5),
            (x, weight[:1], 1e-5),
            (x.double(), weight.double(), 1e-5),
            (x, weight, 0),
            (x, weight, float("inf")),
            (x, weight, 1e-50),
            (torch.ones(0, device="cuda"), torch.ones(0, device="cuda"), 1e-5),
            (torch.ones(65537, device="cuda"), torch.ones(65537, device="cuda"), 1e-5),
        ]
        for args in invalid:
            with pytest.raises(RuntimeError):
                raw(*args)


def test_gradients_use_reference_or_fail_strictly():
    with torch.enable_grad():
        x = torch.ones(3, 33, device="cuda", requires_grad=True)
        weight = torch.ones(33, device="cuda", requires_grad=True)
        assert explain_dispatch(x, weight, 1e-5).reason == "active_gradients"
        rms_norm(x, weight, 1e-5).sum().backward()
        assert x.grad is not None and weight.grad is not None
        with pytest.raises(RuntimeError, match="active_gradients"):
            rms_norm(x, weight, 1e-5, backend="cuda")
        with pytest.raises(RuntimeError, match="inference-only"):
            torch.ops.aegis_norm.rms_norm(x, weight, 1e-5)
