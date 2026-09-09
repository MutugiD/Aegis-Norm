import hashlib
import json

import pytest

from aegis_norm import preflight
from aegis_norm.benchmarks import kernel


@pytest.mark.parametrize("failure", ["blocked", "interrupted", "case_failed", "completed"])
@pytest.mark.parametrize("comparison", ["eager", "geometry128"])
def test_run_status_integrity_and_partial_recovery(tmp_path, monkeypatch, failure, comparison):
    import aegis_norm
    from aegis_norm import experiments

    monkeypatch.setattr(
        preflight,
        "command",
        lambda cmd: {"returncode": 0, "output": "a" * 40 if "rev-parse" in cmd else ""},
    )
    monkeypatch.setattr(preflight, "dependency_snapshot", lambda: "fixture==1\n")
    monkeypatch.setattr(
        preflight, "collect", lambda: {"blockers": ["no T4"] if failure == "blocked" else []}
    )
    monkeypatch.setattr(aegis_norm, "load_native", lambda: None)
    candidate_loads = []
    monkeypatch.setattr(experiments, "load_128", lambda: candidate_loads.append(True))
    calls = []

    def fake_case(case, *, seed, trials, emit, comparison):
        calls.append(case["case_id"])
        emit("correctness", {"case_id": case["case_id"], "passed": True})
        if len(calls) == 2:
            if failure == "interrupted":
                raise KeyboardInterrupt()
            if failure == "case_failed":
                raise RuntimeError("fixture failure")
        return {"case_id": case["case_id"], "fixture": True}

    monkeypatch.setattr(kernel, "run_case", fake_case)
    if failure == "completed":
        kernel.run(tmp_path, trials=2, comparison=comparison)
    else:
        with pytest.raises(KeyboardInterrupt if failure == "interrupted" else RuntimeError):
            kernel.run(tmp_path, trials=2, comparison=comparison)
    (directory,) = tmp_path.iterdir()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["configuration"]["arm_operators"] == kernel.arm_labels(comparison)
    assert len(candidate_loads) == int(comparison == "geometry128" and failure != "blocked")
    assert manifest["status"] == ("failed" if failure in ("blocked", "case_failed") else failure)
    assert manifest["finished_at"]
    summary = json.loads((directory / "summary.json").read_text())
    assert len(summary["cases"]) == (
        0 if failure == "blocked" else 6 if failure == "completed" else 1
    )
    if failure != "completed":
        assert summary["claim_status"] == "not_measured"
    for line in (directory / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest


def test_invalid_configuration_does_not_create_run(tmp_path):
    with pytest.raises(ValueError):
        kernel.run(tmp_path, trials=0)
    assert not list(tmp_path.iterdir())


def test_json_rejects_nonfinite_payload(tmp_path):
    with pytest.raises(ValueError):
        kernel.write_json(tmp_path / "result.json", {"time": float("nan")})
    assert not (tmp_path / "result.json").exists()
