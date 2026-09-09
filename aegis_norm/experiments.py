"""Explicit experimental kernels; never selected by the production dispatcher."""

import os
import threading
from pathlib import Path

_loaded128 = False
_lock = threading.Lock()


def load_128(*, verbose=True):
    """Compile the candidate separately on T4; restart after a partial load failure."""
    global _loaded128
    with _lock:
        if _loaded128:
            return
        from .native import load_native

        load_native(verbose=verbose)
        import torch
        from torch.utils.cpp_extension import load

        from .registration import fake_rms_norm

        if hasattr(torch.ops.aegis_norm_experiments, "rms_norm_128"):
            raise RuntimeError("128-thread operator is already registered by another loader")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
        os.environ["MAX_JOBS"] = "2"
        sources = Path(__file__).parent / "csrc"
        load(
            name="aegis_rmsnorm128_experiment",
            sources=[str(sources / "rmsnorm128_binding.cpp"), str(sources / "rmsnorm128_cuda.cu")],
            extra_cflags=["-O2"],
            extra_cuda_cflags=["-O2", "-lineinfo"],
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )
        torch.library.register_fake("aegis_norm_experiments::rms_norm_128", fake_rms_norm)
        _loaded128 = True
