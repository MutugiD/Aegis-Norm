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


@pytest.mark.parametrize("registration_fails", [False, True])
def test_loader_registers_metadata_after_schema_before_ready(monkeypatch, registration_fails):
    from types import SimpleNamespace

    import torch
    import torch.utils.cpp_extension

    from aegis_norm import preflight, registration

    monkeypatch.setattr(native, "_loaded", False)
    monkeypatch.setattr(preflight, "collect", lambda: {"blockers": []})
    namespace = SimpleNamespace()
    monkeypatch.setattr(torch, "ops", SimpleNamespace(aegis_norm=namespace))
    # Restore loader environment changes after this test.
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "fixture")
    monkeypatch.setenv("MAX_JOBS", "fixture")
    calls = []

    def build(**kwargs):
        calls.append("build")
        assert not native.is_loaded()
        namespace.rms_norm = object()

    def register():
        calls.append("fake")
        assert hasattr(namespace, "rms_norm") and not native.is_loaded()
        if registration_fails:
            raise RuntimeError("fixture registration failure")

    monkeypatch.setattr(torch.utils.cpp_extension, "load", build)
    monkeypatch.setattr(registration, "register_fake", register)
    if registration_fails:
        with pytest.raises(RuntimeError, match="fixture registration failure"):
            native.load_native()
        assert not native.is_loaded()
    else:
        native.load_native()
        native.load_native()
        assert native.is_loaded()
    assert calls == ["build", "fake"]
