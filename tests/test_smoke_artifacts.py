import hashlib
import json

from aegis_norm import smoke


def test_blocked_attempt_exports_honest_evidence_and_never_builds(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aegis-smoke", "--output-root", str(tmp_path)])
    monkeypatch.setattr(smoke, "collect", lambda: {"blockers": ["No T4"], "status": "not_ready"})
    monkeypatch.setattr(smoke, "dependency_snapshot", lambda: "torch==2.14.0+cpu\n")
    monkeypatch.setattr(smoke, "command", lambda args: {"returncode": 0, "output": "fixture"})

    def unexpected_build(*args, **kwargs):
        raise AssertionError("A blocked preflight must not start a native build")

    monkeypatch.setattr(smoke.subprocess, "run", unexpected_build)
    assert smoke.main() == 1
    attempt = next(tmp_path.iterdir())
    result = json.loads((attempt / "result.json").read_text())
    assert result["status"] == "blocked"
    assert result["performance"] == "not_measured"
    assert not (attempt / "checks.json").exists()
    hashes = json.loads((attempt / "sha256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((attempt / name).read_bytes()).hexdigest() == expected
    assert smoke.main() == 1
    assert len(list(tmp_path.iterdir())) == 2


def test_worker_failure_cannot_be_reported_as_pass(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aegis-smoke", "--output-root", str(tmp_path)])
    monkeypatch.setattr(smoke, "collect", lambda: {"blockers": [], "status": "ready_for_build"})
    monkeypatch.setattr(smoke, "dependency_snapshot", lambda: "torch==2.14.0+cu126\n")
    monkeypatch.setattr(smoke, "command", lambda args: {"returncode": 0, "output": "fixture"})
    monkeypatch.setattr(
        smoke.subprocess, "run", lambda *args, **kwargs: smoke.subprocess.CompletedProcess(args, 1)
    )
    assert smoke.main() == 1
    result = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert result["status"] == "failed" and result["returncode"] == 1
