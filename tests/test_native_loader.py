import pytest

from aegis_norm import native


def test_loader_stops_before_compilation_on_preflight_failure(monkeypatch):
    from aegis_norm import preflight

    monkeypatch.setattr(native, "_loaded", False)
    monkeypatch.setattr(preflight, "collect", lambda: {"blockers": ["fixture: missing T4"]})
    with pytest.raises(RuntimeError, match="missing T4"):
        native.load_native()
    assert not native.is_loaded()


def test_loader_is_idempotent_after_success(monkeypatch):
    from aegis_norm import preflight

    monkeypatch.setattr(native, "_loaded", True)

    def unexpected_preflight():
        raise AssertionError("An already registered operator must not compile again")

    monkeypatch.setattr(preflight, "collect", unexpected_preflight)
    native.load_native()
