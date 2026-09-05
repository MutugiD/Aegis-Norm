"""Report the actual notebook environment before attempting a native build."""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def command(args):
    """Run a bounded diagnostic without shell interpolation."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return {"returncode": result.returncode, "output": result.stdout + result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "output": str(error)}


def blockers(report):
    """C2 candidate policy; passing this is permission to try a build, not a pass."""
    reasons = []
    if report["system"] != "Linux" or report["machine"] not in ("x86_64", "AMD64"):
        reasons.append("Required native target is Linux x86_64 in a T4 notebook")
    if report["python_minor"] not in ([3, 11], [3, 12], [3, 13]):
        reasons.append("C2 candidate expects Python 3.11, 3.12 or 3.13")
    torch_info = report["torch"]
    if torch_info.get("version", "").split("+")[0] != "2.14.0":
        reasons.append("C2 candidate requires torch 2.14.0; record/review alternatives")
    if torch_info.get("cuda_runtime") != "12.6":
        reasons.append("C2 candidate requires the cu126 PyTorch wheel")
    if not torch_info.get("cuda_available"):
        reasons.append("PyTorch cannot access CUDA; select a GPU runtime and inspect driver")
    device = report.get("device", {})
    if device.get("capability") != [7, 5] or "T4" not in device.get("name", ""):
        reasons.append("Required qualification device is an NVIDIA T4 (capability 7.5)")
    if report["nvcc"]["returncode"] != 0:
        reasons.append("nvcc is missing or failed; a PyTorch wheel is not a CUDA toolkit")
    elif not re.search(r"release 12\.6\b", report["nvcc"]["output"]):
        reasons.append("C2 candidate requires CUDA toolkit 12.6; review toolkit mismatch")
    if report["compiler"]["returncode"] != 0:
        reasons.append("A working C++ host compiler is required in the notebook VM")
    if report["ninja"]["returncode"] != 0:
        reasons.append("Ninja is missing; install the package native extra")
    if report.get("torch_error"):
        reasons.append("PyTorch/device inspection failed; inspect torch_error")
    return reasons


def collect():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda_home) / "bin" / "nvcc") if cuda_home else shutil.which("nvcc")
    if not nvcc and Path("/usr/local/cuda/bin/nvcc").is_file():
        nvcc = "/usr/local/cuda/bin/nvcc"
    report = {
        "schema_version": 1,
        "candidate": "C2-torch214-cu126",
        "qualification": "not_run",
        "python": platform.python_version(),
        "python_minor": list(sys.version_info[:2]),
        "system": platform.system(),
        "machine": platform.machine(),
        "cuda_home": cuda_home,
        "nvcc_path": nvcc,
        "nvcc": command([nvcc or "nvcc", "--version"]),
        "compiler": command(shlex.split(os.environ.get("CXX", "c++")) + ["--version"]),
        "ninja": command(["ninja", "--version"]),
        "driver": command(["nvidia-smi"]),
        "torch": {},
        "build": {"architecture": "7.5", "max_jobs": 2, "fast_math": False},
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cxx11_abi": torch.compiled_with_cxx11_abi(),
        }
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(index)
            report["device"] = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "free_bytes": free,
                "total_bytes": total,
                "wheel_architectures": torch.cuda.get_arch_list(),
            }
    except (ImportError, RuntimeError, OSError) as error:
        report["torch_error"] = str(error)
    report["blockers"] = blockers(report)
    report["status"] = "not_ready" if report["blockers"] else "ready_for_build"
    return report


def dependency_snapshot():
    """Versions only: avoid exporting credential-bearing pip direct URLs."""
    return (
        "\n".join(
            sorted(
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-t4", action="store_true")
    args = parser.parse_args()
    report = collect()
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return int(args.require_t4 and bool(report["blockers"]))


if __name__ == "__main__":
    raise SystemExit(main())
