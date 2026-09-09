"""Exercise real timing and artifact records in the T4 notebook."""

import os

import pytest
import torch

from aegis_norm import load_native
from aegis_norm.benchmarks import kernel
from aegis_norm.benchmarks.kernel import run_case

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("AEGIS_RUN_GPU") != "1", reason="T4 notebook opt-in required"
    ),
]


@pytest.mark.parametrize("comparison", ["eager", "geometry128"])
def test_real_paired_operator_measurement(comparison):
    load_native()
    if comparison == "geometry128":
        from aegis_norm.experiments import load_128

        load_128()
    records = []
    case = {
        "case_id": "measurement-test",
        "rows": 2,
        "hidden": 257,
        "dtype": "float16",
        "eps": 1e-5,
        "offset": 1,
    }
    with torch.inference_mode():
        result = run_case(
            case,
            seed=2026,
            comparison=comparison,
            trials=2,
            emit=lambda name, row: records.append((name, row)),
        )
    samples = [row for name, row in records if name == "samples"]
    assert len(samples) == 4
    assert len({s["repetitions"] for s in samples}) == 1
    assert all(s["elapsed_ms"] > 0 and s["host_elapsed_ms"] > 0 for s in samples)
    assert result["correctness_passed"]
    assert records[0][0] == "correctness" and records[0][1]["passed"]


@pytest.mark.parametrize("comparison", ["eager", "geometry128"])
def test_numerical_failure_prevents_timing(monkeypatch, comparison):
    load_native()
    if comparison == "geometry128":
        from aegis_norm.experiments import load_128

        load_128()
    monkeypatch.setattr(kernel, "reference", lambda x, weight, eps: torch.zeros_like(x))
    records = []
    case = {
        "case_id": "bad-baseline",
        "rows": 2,
        "hidden": 257,
        "dtype": "float16",
        "eps": 1e-5,
        "offset": 0,
    }
    with torch.inference_mode(), pytest.raises(RuntimeError, match="Numerical gate failed"):
        run_case(
            case,
            seed=2026,
            comparison=comparison,
            trials=2,
            emit=lambda name, row: records.append((name, row)),
        )
    assert [name for name, _ in records] == ["correctness"]
    assert not records[0][1]["passed"]
