"""RMSNorm reference, explicit CUDA loading and notebook build tools."""

__version__ = "0.1.0.dev1"
__all__ = ["rms_norm", "explain_dispatch", "load_native", "__version__"]


def __getattr__(name):
    if name == "load_native":
        from .native import load_native

        return load_native
    if name in ("rms_norm", "explain_dispatch"):
        from .ops import rmsnorm

        return getattr(rmsnorm, name)
    raise AttributeError(name)
