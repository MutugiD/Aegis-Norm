import copy
import json
import subprocess
import sys

import pytest

from aegis_norm.preflight import blockers, command


def candidate():
    return {
        "system": "Linux",
        "machine": "x86_64",
        "python_minor": [3, 11],
        "torch": {"version": "2.14.0+cu126", "cuda_runtime": "12.6", "cuda_available": True},
        "device": {"name": "Tesla T4", "capability": [7, 5]},
        "nvcc": {"returncode": 0, "output": "Cuda compilation tools, release 12.6, V12.6.85"},
        "compiler": {"returncode": 0},
        "ninja": {"returncode": 0},
    }


def test_candidate_allows_attempt_but_is_not_gpu_evidence():
    assert blockers(candidate()) == []


@pytest.mark.parametrize(
    "field,replacement,expected",
    [
        ("device", {"name": "A100", "capability": [8, 0]}, "T4"),
        ("nvcc", {"returncode": None, "output": "missing"}, "nvcc"),
        ("nvcc", {"returncode": 0, "output": "release 13.0"}, "toolkit mismatch"),
        ("compiler", {"returncode": 1}, "host compiler"),
        ("ninja", {"returncode": None}, "Ninja"),
        ("torch", {"cuda_available": False}, "cannot access CUDA"),
        ("system", "Windows", "Linux"),
    ],
)
def test_environment_failures_are_actionable(field, replacement, expected):
    report = copy.deepcopy(candidate())
    report[field] = replacement
    assert any(expected in reason for reason in blockers(report))


def test_missing_command_is_reported():
    assert command(["aegis-nonexistent-tool-82736"])["returncode"] is None


def test_package_import_does_not_load_torch_or_extension():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import aegis_norm, sys, json; print(json.dumps(list(sys.modules)))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    modules = json.loads(result.stdout)
    assert "torch" not in modules
    assert "torch.utils.cpp_extension" not in modules
