import hashlib
import json

import pytest

from aegis_norm import smoke


def test_blocked_attempt_exports_honest_evidence_and_never_builds(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aegis-smoke", "--output-root", str(tmp_path)])
    monkeypatch.setattr(smoke, "collect", lambda: {"blockers": ["No T4"], "status": "not_ready"})
    monkeypatch.setattr(smoke, "dependency_snapshot", lambda: "torch==2.14.0+cpu\n")
    monkeypatch.setattr(smoke, "command", lambda args: {"returncode": 0, "output": "fixture"})

    def unexpected_build(*args, **kwargs):
        raise AssertionError("A blocked preflight must not start a native build")

    monkeypatch.setattr(smoke, "execute_worker", unexpected_build)
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
    monkeypatch.setattr(smoke, "execute_worker", lambda *args, **kwargs: 1)
    assert smoke.main() == 1
    result = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert result["status"] == "failed" and result["returncode"] == 1


def test_zero_exit_without_check_evidence_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aegis-smoke", "--output-root", str(tmp_path)])
    monkeypatch.setattr(smoke, "collect", lambda: {"blockers": []})
    monkeypatch.setattr(smoke, "dependency_snapshot", lambda: "fixture\n")
    monkeypatch.setattr(smoke, "command", lambda args: {})
    monkeypatch.setattr(smoke, "execute_worker", lambda *args, **kwargs: 0)
    assert smoke.main() == 1
    result = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert result["status"] == "failed"


def test_worker_uses_fresh_cache_and_records_success(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["aegis-smoke", "--output-root", str(tmp_path)])
    monkeypatch.setattr(smoke, "collect", lambda: {"blockers": []})
    monkeypatch.setattr(smoke, "dependency_snapshot", lambda: "fixture\n")
    monkeypatch.setattr(smoke, "command", lambda args: {})

    def worker(output, env):
        assert env["TORCH_EXTENSIONS_DIR"]
        smoke.write_json(output / "checks.json", {"status": "passed", "cases": list(range(8))})
        return 0

    monkeypatch.setattr(smoke, "execute_worker", worker)
    assert smoke.main() == 0
    result = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert result["build_cache"] == "fresh" and result["status"] == "passed"


@pytest.mark.parametrize(
    "error", [smoke.subprocess.TimeoutExpired("worker", 900), KeyboardInterrupt()]
)
def test_timeout_and_interrupt_stop_worker_tree(tmp_path, monkeypatch, error):
    events = []

    class Child:
        pid = 12345

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is not None:
                raise error
            return -9

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: Child())
    if smoke.os.name == "posix":
        monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: events.append(("killpg", pid)))
    with pytest.raises(type(error)):
        smoke.execute_worker(tmp_path, {})
    assert events[0] == ("wait", 900) and events[-1] == ("wait", None)
    assert events[1] == (("killpg", 12345) if smoke.os.name == "posix" else "kill")
