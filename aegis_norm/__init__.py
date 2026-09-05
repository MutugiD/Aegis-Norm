"""Reference RMSNorm and explicit notebook build tools; native RMSNorm is pending."""

__version__ = "0.1.0.dev0"
__all__ = ["rms_norm", "explain_dispatch", "__version__"]


def __getattr__(name):
    if name in ("rms_norm", "explain_dispatch"):
        from .ops import rmsnorm

        return getattr(rmsnorm, name)
    raise AttributeError(name)
