import pytest

from aegis_norm import experiments, native


def test_experiment_preflight_failure_stops_build(monkeypatch):
    monkeypatch.setattr(experiments, "_loaded128", False)

    def blocked(**kwargs):
        raise RuntimeError("fixture preflight blocker")

    monkeypatch.setattr(native, "load_native", blocked)
    with pytest.raises(RuntimeError, match="preflight blocker"):
        experiments.load_128()
    assert not experiments._loaded128


def test_loaded_experiment_is_idempotent(monkeypatch):
    monkeypatch.setattr(experiments, "_loaded128", True)

    def unexpected(**kwargs):
        raise AssertionError("Unexpected load")

    monkeypatch.setattr(native, "load_native", unexpected)
    experiments.load_128()


@pytest.mark.parametrize("fails", [False, True])
def test_experiment_registration_order_and_failure(monkeypatch, fails):
    from types import SimpleNamespace

    import torch
    import torch.utils.cpp_extension

    namespace = SimpleNamespace()
    calls = []
    monkeypatch.setattr(experiments, "_loaded128", False)
    monkeypatch.setattr(native, "load_native", lambda **kwargs: calls.append("baseline"))
    monkeypatch.setattr(torch, "ops", SimpleNamespace(aegis_norm_experiments=namespace))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "fixture")
    monkeypatch.setenv("MAX_JOBS", "fixture")

    def build(**kwargs):
        assert all("rmsnorm128" in source for source in kwargs["sources"])
        calls.append("candidate")
        namespace.rms_norm_128 = object()

    def register(name, function):
        assert name == "aegis_norm_experiments::rms_norm_128"
        assert hasattr(namespace, "rms_norm_128") and not experiments._loaded128
        calls.append("fake")
        if fails:
            raise RuntimeError("fixture metadata failure")

    monkeypatch.setattr(torch.utils.cpp_extension, "load", build)
    monkeypatch.setattr(torch.library, "register_fake", register)
    if fails:
        with pytest.raises(RuntimeError, match="metadata failure"):
            experiments.load_128()
        assert not experiments._loaded128
    else:
        experiments.load_128()
        experiments.load_128()
        assert experiments._loaded128
    assert calls == ["baseline", "candidate", "fake"]


def test_candidate_operator_labels_cannot_mix_with_eager():
    from aegis_norm.benchmarks.kernel import arm_labels

    assert arm_labels("geometry128") == {
        "reference": "aegis_norm::rms_norm",
        "native": "aegis_norm_experiments::rms_norm_128",
    }
    with pytest.raises(ValueError):
        arm_labels("unknown")
