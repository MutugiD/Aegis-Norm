"""Explicitly opt in inside the T4 notebook; a skip is not GPU qualification."""

import os

import pytest
import torch

from aegis_norm.smoke import build_smoke, run_checks

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("AEGIS_RUN_GPU") != "1", reason="T4 notebook opt-in required"
    ),
]


@pytest.fixture(scope="module")
def extension():
    return build_smoke()


def test_native_execution_and_cuda_reference(extension):
    assert len(run_checks(extension)) == 8


def test_native_binding_rejects_invalid_arguments(extension):
    with pytest.raises(RuntimeError, match="CUDA"):
        extension.forward(torch.ones(2))
    with pytest.raises(RuntimeError, match="FP32"):
        extension.forward(torch.ones(2, device="cuda", dtype=torch.float16))
    with pytest.raises(RuntimeError, match="contiguous"):
        extension.forward(torch.ones(2, 3, device="cuda").t())
    with torch.enable_grad(), pytest.raises(RuntimeError, match="inference-only"):
        extension.forward(torch.ones(2, device="cuda", requires_grad=True))
