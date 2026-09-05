"""Explicit compilation/registration of the inference-only CUDA RMSNorm operator."""

import argparse
import os
import threading
from pathlib import Path

_loaded = False
_lock = threading.Lock()


def is_loaded():
    return _loaded


def load_native(*, verbose=True):
    """Build/load once per process; call explicitly inside the qualified notebook.

    Source changes after loading require a new Python process. No public operator
    call implicitly starts compilation. Compilation errors propagate unchanged.
    """
    global _loaded
    with _lock:
        if _loaded:
            return
        from .preflight import collect

        report = collect()
        if report["blockers"]:
            raise RuntimeError("Native RMSNorm preflight failed: " + "; ".join(report["blockers"]))
        import torch
        from torch.utils.cpp_extension import load

        if hasattr(torch.ops.aegis_norm, "rms_norm"):
            raise RuntimeError("aegis_norm::rms_norm is already registered by another loader")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        os.environ["MAX_JOBS"] = "2"
        sources = Path(__file__).parent / "csrc"
        load(
            name="aegis_rmsnorm_native",
            sources=[str(sources / "rmsnorm_binding.cpp"), str(sources / "rmsnorm_cuda.cu")],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2", "-lineinfo"],
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )
        if not hasattr(torch.ops.aegis_norm, "rms_norm"):
            raise RuntimeError("Native build did not register aegis_norm::rms_norm")
        _loaded = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    load_native()
    print("Native RMSNorm registered; numerical validation remains a separate test step")


if __name__ == "__main__":
    raise SystemExit(main())
