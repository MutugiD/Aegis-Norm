# F01: package, reference and T4 build walkthrough

Status: code implemented; initial T4 smoke tests passed, complete artifact qualification pending. See the [contributor evidence review](../evaluation/05-f01-gpu-evidence-review.md). This increment addresses the F01 portion of REQ-K01 and REQ-K05, with initial T-K01/T-N01/T-N02 coverage. It does not complete F02 or establish RMSNorm speed.

## 1. What the package contains

The [package metadata](../../pyproject.toml) tells pip how to build a source distribution and a wheel. A wheel is an installable Python distribution. This wheel contains Python modules and the native source files, not a precompiled CUDA binary. The separate explicit smoke command compiles those sources inside the notebook session. No local CUDA installation is required.

The Python import exposes `rms_norm` and `explain_dispatch`. Importing the top-level package does not invoke a compiler. `rms_norm` currently executes the reference implementation; requesting `backend="cuda"` raises a clear error because native RMSNorm belongs to F02. The smoke extension has a separate name and is never selected as RMSNorm.

## 2. Reference RMSNorm and precision

The [reference implementation](../../aegis_norm/ops/rmsnorm.py) calculates:

```python
x32 = x.float()
inverse_rms = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
normalized = (x32 * inverse_rms).to(x.dtype)
output = weight * normalized
```

The final dimension contains the features of each token row. `keepdim=True` retains a size-one axis so that one normalization scale broadcasts across a row. The reference uses FP32 for the squared values, reduction and scale calculation. Casting normalized values before multiplying by the weight preserves the selected Llama rounding boundary. Mixed weight dtypes retain PyTorch promotion behavior. PyTorch autograd remains available for the reference.

These are tensor operations: a CUDA input stays on CUDA. Reference does not mean CPU. CPU tests are supplementary checks of semantics and errors; required notebook tests run the reference with T4 tensors.

`explain_dispatch(x, weight, eps)` reports `reference` with reason `native_rmsnorm_not_implemented` in auto mode. Invalid shapes, devices, dtype categories and epsilon values fail before execution; fallback does not repair malformed inputs.

## 3. Preflight: describe before building

The [preflight command](../../aegis_norm/preflight.py) reports the driver, wheel CUDA runtime, toolkit compiler, host compiler, Ninja, Python, GPU capability and memory. These are separate dependencies:

| Concept | Responsibility |
|---|---|
| NVIDIA driver | Provides OS access to the attached GPU |
| CUDA toolkit / nvcc | Compiles CUDA source on the notebook VM's CPU |
| C++ compiler | Compiles the host binding and launcher code |
| PyTorch CUDA wheel | Provides PyTorch and its selected CUDA runtime dependencies |
| Ninja | Coordinates native compilation tasks |
| T4 / architecture 7.5 | Executes the generated GPU instructions |

The C2 candidate is Linux x86_64, Python 3.11-3.13, PyTorch 2.14.0/cu126 and CUDA toolkit 12.6. Exact compiler and driver versions remain runtime evidence; preflight is not an exhaustive ABI/compiler compatibility test. `ready_for_build` only allows an attempt. A missing tool, different GPU or candidate mismatch is a blocker with an explanation. Alternative environments need compatibility review; the tool does not modify drivers or toolkits.

## 4. The Python-to-CUDA bridge

The [C++ binding](../../aegis_norm/csrc/smoke_binding.cpp) accepts a PyTorch tensor through pybind11. It validates CUDA placement, FP32 dtype, contiguous layout and inference-only usage. A tensor contains metadata and a pointer to its storage. `data_ptr<float>()` is appropriate here because the binding requires FP32; it would be incorrect for FP16 storage.

The [CUDA launcher](../../aegis_norm/csrc/smoke_cuda.cu) uses a device guard, allocates the output through PyTorch and launches `y = 3 * x`. A grid contains blocks; each block contains 256 threads. A grid-stride loop lets threads cover arbitrary vector lengths, including odd tails. An empty tensor returns without launching. Input and output use separate storage.

The kernel runs on PyTorch's current stream for that device. A stream orders GPU work; returning from the host function does not wait for that work to complete. The launcher checks launch errors without a global synchronization. Tests wait before inspecting results and include a non-default stream. This smoke is an initial ordering check, not the complete T-K06 suite or sanitizer qualification.

## 5. Build and run in the notebook

Open [the T4 notebook](../../notebooks/01-t4-build-smoke.ipynb), enable a T4 and internet access, and set `REVISION` to the feature PR's full commit SHA. Run its cells in order:

1. Check out that commit in a new notebook workspace.
2. Inspect `nvidia-smi`, `nvcc` and the C++ compiler before downloading dependencies.
3. Create an isolated Python environment; explicitly install the cu126 wheel and pinned foundation dependencies, then the package.
4. Run preflight, compile the smoke extension and execute native/reference checks.
5. Run the reference, preflight and opted-in GPU binding tests.
6. Export evidence even if an earlier cell failed.

The installed commands, executed with the notebook environment active, are:

```text
python -m aegis_norm.preflight --require-t4 --output artifacts/preflight.json
python -m aegis_norm.smoke --output-root artifacts
```

The smoke compiler uses two host jobs, architecture 7.5, optimization level O2 and no global fast-math flag. PyTorch caches the native extension for that environment. A cache hit can test loading and execution but does not replace the required fresh-session compilation evidence.

## 6. Evidence and dependency resolution

Each smoke attempt gets a unique directory. It contains preflight metadata, resolved package versions, the tested commit/worktree status, native source hashes, a status record and artifact hashes. If building starts, compiler output is retained in `build.log`. Successful execution produces `checks.json`. Statuses distinguish blocked, running, passed, failed, timed out and interrupted; a forcibly ended session can leave `running`, which is incomplete evidence.

[Foundation requirements](../../requirements-foundation.txt) pin direct dependencies. The session's `environment-versions.txt` records resolved versions without exporting private package URLs. It is an inventory, not a portable, hash-locked T4 environment. A qualified reproduction lock remains pending until a real T4 run is reviewed. Keep the wheel index and compiler/driver metadata with that inventory.

The standalone CLI records the current working directory's Git identity; run it from the tested checkout as the notebook does. The notebook installs that checkout before invoking the CLI. Do not attribute an installed package from another commit to the current checkout. The smoke timeout bounds its worker; a hard notebook interruption can require a fresh session before retrying. Full interruption/recovery handling follows in F06.

## 7. What the tests establish

Supporting tests cover known values, the FP16 cast boundary, odd widths, empty tensors, noncontiguous reference inputs, mixed dtypes, autograd, invalid metadata/epsilon, special values, dispatch explanations, preflight failures and notebook Python syntax. Package CI builds both distributions, installs the wheel, verifies included native sources and audits resolved dependencies. GPU tests are explicitly deselected in CPU CI.

The notebook smoke checks native output and input preservation, empty/odd vectors, a side stream and the reference on FP16/FP32 CUDA tensors. Additional GPU tests reject invalid binding inputs. None of these results may be reported as native RMSNorm correctness, model acceleration or benchmark evidence. The first contributor run passed 38 tests including both GPU cases, but omitted standalone build metadata; F01 remains unqualified until the missing evidence and reproduction gate are resolved.

## Sources and candidate change

Verified 2026-09-05: [PyTorch extension build requirements](https://docs.pytorch.org/docs/main/cpp_extension.html), [CUDA streams](https://docs.pytorch.org/cppdocs/api/cuda/streams.html), [NVIDIA nvcc](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html), [setuptools package data](https://setuptools.pypa.io/en/latest/userguide/datafiles.html), and [PyTorch 2.14 release](https://pytorch.org/blog/pytorch-2-14-release-blog/).

The earlier 2.8.0 candidate produced eight known-vulnerability findings in the direct package audit. The 2.14.0 direct audit found none at review time; full resolved-environment audits run separately. C2 replaces the old F01 installation candidate without claiming GPU support has been measured. Transformers/model compatibility remains a later F05 qualification, not inherited from this package upgrade.
