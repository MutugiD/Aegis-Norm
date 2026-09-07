"""T-K09: actual dispatcher/FakeTensor/AOT checks, executed on contributor T4."""

import os

import pytest
import torch

from aegis_norm import load_native, rms_norm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("AEGIS_RUN_GPU") != "1", reason="T4 notebook opt-in required"
    ),
]


@pytest.fixture(scope="module", autouse=True)
def native_operator():
    load_native()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(7,), (2, 257), (2, 3, 32), (0, 7)])
def test_opcheck(dtype, shape):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    with torch.no_grad():
        result = torch.library.opcheck(torch.ops.aegis_norm.rms_norm.default, (x, weight, 1e-5))
    assert len(result) == 4 and all(value == "SUCCESS" for value in result.values())


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dynamic_aot_eager_with_offset_input(dtype):
    # aot_eager validates graph execution without requesting a Triton kernel.
    def forward(x, weight):
        return torch.ops.aegis_norm.rms_norm.default(x, weight, 1e-5)

    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=True)
    with torch.no_grad():
        for rows, width in [(2, 7), (3, 257), (4, 7)]:
            x = torch.randn(rows * width + 1, device="cuda", dtype=dtype)[1:].view(rows, width)
            weight = torch.randn(width, device="cuda", dtype=dtype)
            actual = compiled(x, weight)
            expected = rms_norm(x, weight, 1e-5, backend="reference")
            atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (1e-6, 2e-5)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.cuda.synchronize()
