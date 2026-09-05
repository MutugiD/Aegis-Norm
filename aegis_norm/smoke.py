"""Explicit native build-and-run check; this does not implement CUDA RMSNorm."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .preflight import collect, command, dependency_snapshot


def build_smoke():
    """Compile on the notebook host and load a pybind11 extension into Python."""
    report = collect()
    if report["blockers"]:
        raise RuntimeError("Native smoke preflight failed: " + "; ".join(report["blockers"]))
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["MAX_JOBS"] = "2"
    from torch.utils.cpp_extension import load

    sources = Path(__file__).parent / "csrc"
    return load(
        name="aegis_build_smoke",
        sources=[str(sources / "smoke_binding.cpp"), str(sources / "smoke_cuda.cu")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "-lineinfo"],
        with_cuda=True,
        verbose=True,
    )


def run_checks(extension):
    import torch

    from . import rms_norm

    cases = []
    with torch.inference_mode():
        for count in (0, 1, 7, 257, 4097):
            x = torch.arange(count, device="cuda", dtype=torch.float32)
            before = x.clone()
            y = extension.forward(x)
            torch.testing.assert_close(y, x * 3, atol=0, rtol=0)
            torch.testing.assert_close(x, before, atol=0, rtol=0)
            if count:
                assert y.data_ptr() != x.data_ptr()
            cases.append(f"native_triple_{count}")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.empty(4097, device="cuda").fill_(7)
            y = extension.forward(x)
            consumed = y + 1
        stream.synchronize()  # Completion belongs in the test, not the launcher.
        torch.testing.assert_close(consumed, torch.full_like(consumed, 22), atol=0, rtol=0)
        cases.append("native_side_stream")
        for dtype in (torch.float16, torch.float32):
            x = torch.tensor([[3, 4], [0, 0]], device="cuda", dtype=dtype)
            weight = torch.tensor([2, -1], device="cuda", dtype=dtype)
            y = rms_norm(x, weight, 1e-5, backend="reference")
            expected = torch.tensor(
                [[6 / (12.5 + 1e-5) ** 0.5, -4 / (12.5 + 1e-5) ** 0.5], [0, 0]],
                device="cuda",
                dtype=dtype,
            )
            torch.testing.assert_close(y, expected, atol=2e-3, rtol=2e-3)
            cases.append(f"cuda_reference_{dtype}")
    torch.cuda.synchronize()
    return cases


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        cases = run_checks(build_smoke())
        write_json(args.worker / "checks.json", {"status": "passed", "cases": cases})
        return 0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    report = collect()
    write_json(output / "preflight.json", report)
    (output / "environment-versions.txt").write_text(dependency_snapshot(), encoding="utf-8")
    sources = Path(__file__).parent / "csrc"
    result = {
        "schema_version": 1,
        "run_id": run_id,
        "package_version": __version__,
        "commit": command(["git", "rev-parse", "HEAD"]),
        "worktree": command(["git", "status", "--porcelain"]),
        "status": "not_run",
        "rmsnorm_native": "not_implemented",
        "performance": "not_measured",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources.iterdir())
        },
    }
    write_json(output / "result.json", result)  # Survives interrupted compilation.
    if report["blockers"]:
        result["status"] = "blocked"
        result["blockers"] = report["blockers"]
    else:
        result["status"] = "running"
        write_json(output / "result.json", result)
        try:
            with (output / "build.log").open("w", encoding="utf-8") as log:
                child = subprocess.run(
                    [sys.executable, "-m", "aegis_norm.smoke", "--worker", str(output.resolve())],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                    check=False,
                )
            result["status"] = "passed" if child.returncode == 0 else "failed"
            result["returncode"] = child.returncode
        except subprocess.TimeoutExpired:
            result["status"] = "timed_out"
        except KeyboardInterrupt:
            result["status"] = "interrupted"
        except OSError as error:
            result["status"] = "failed"
            result["error"] = str(error)
    write_json(output / "result.json", result)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    write_json(output / "sha256.json", hashes)
    print(json.dumps({"status": result["status"], "artifacts": str(output)}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
