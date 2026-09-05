# Contributor and notebook workflow

Status: F01 package, preflight and initial build notebook implemented; required GPU execution pending. Use the [F01 walkthrough](../implementation/01-foundation-walkthrough.md) for executable steps. The full model/evaluation workflow below remains F02-F06 work.

## Local development and review

Use the local repository checkout for editing, Git and optional lightweight checks. Build the native extension, run the reference on CUDA tensors, execute CUDA tests and benchmark inside the contributor's Linux Colab/Kaggle T4 session. No local CUDA toolkit, local native compilation or local GPU is required. Read the [compatibility candidate](../research/03-compatibility.md), [operator contract](../architecture/02-operator-contract.md) and [test plan](../testing/01-test-plan.md) before native changes.

Compilation happens on the notebook VM's CPU using `nvcc` and a compatible C++ compiler; the compiled kernel executes on the attached T4. Installing PyTorch alone does not supply the complete extension build toolchain. Preflight must inspect the session before building. See [PyTorch extension build requirements](https://docs.pytorch.org/docs/main/cpp_extension.html) and [NVIDIA's compiler guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html).

F01 supplies the minimal executable setup/preflight/build notebook and a native smoke operation. Verify native execution explicitly and export its logs; reference fallback cannot pass this gate. F02 adds the RMSNorm correctness suite, and F06 expands the notebooks to complete contributor runs. CPU-only import support keeps supporting CI usable without a CUDA compiler; it does not relocate CUDA testing to the local machine or replace T4 evidence. A tunnel, when used, provides access to the remote session; it does not make the T4 a local CUDA device.

Create each feature branch from updated main only after the previous PR is green and merged. Keep no more than one PR open at a time. Use `task/<topic>` or `feat/<topic>` branch names, and `/task: ...` or `/feat: ...` commit subjects. PR titles use `task:` or `feat:` without the slash; descriptions explain context, purpose, changes and completed tests rather than internal workflow rules. Never add generated author trailers, private prompts, tokens, model weights or build outputs. Keep [product-context.md](../product-context.md) readable and exploratory; update research/decisions rather than making its examples appear validated. The original unedited source remains in Git history.

## Required notebook cells

| Order | Cell responsibility | Failure handling |
|---|---|---|
| 1 | Explain profile, provider restrictions and expected resource use | Contributor chooses smoke or required run explicitly |
| 2 | Capture `nvidia-smi`, Python, OS, torch/runtime, nvcc/compiler and free memory | Missing GPU/toolchain or wrong GPU is reported before compilation |
| 3 | Fetch a selected repository commit; record it | No moving-branch benchmark without resolved commit |
| 4 | Create/select a compatible environment and install candidate dependencies | Export resolver/build failures; restart interpreter after changes |
| 5 | Compile native extension for detected qualified architecture | Log build flags/compiler; no silent reference fallback |
| 6 | Run numerical and stream smoke tests | Stop performance section on required failure |
| 7 | Load pinned tokenizer/model, FP16, eager attention; verify adaptation | Record patched count and model memory; OOM stops model section |
| 8 | Run bounded selected benchmark profile, case by case | Write completed cases immediately; optional unavailable baselines are explicit |
| 9 | Run local request functionality and serving workload after F08 exists | No tunnel required; native/reference mode shown in artifacts |
| 10 | Finalize manifest, hashes and export archive | Failed/interrupted attempts export partial evidence with honest status |

Provider notebooks must contain actual executable cells once F06 is implemented, not just links to shell snippets. Colab and Kaggle wrappers may differ in install/export paths but invoke the same package, tests and benchmark harnesses.

## Candidate installation and build rules

F01 supersedes the earlier 2.8.0 wheel selection with candidate C2: `torch==2.14.0` from the official cu126 index and toolkit 12.6, following dependency audit findings. The notebook creates an isolated environment and installs the foundation package without Transformers or model weights. F05 must revalidate its model dependency selection. A qualified environment lock remains pending T4 execution; the smoke exports resolved versions. Match toolkit family and record GCC; do not mistake `nvidia-smi`'s CUDA display for installed nvcc. Source builds use `TORCH_CUDA_ARCH_LIST=7.5` and `MAX_JOBS=2`.

Do not automatically replace a notebook's Python/driver/toolkit to force C2. If C2 cannot run, capture the provider environment and review an alternative candidate before comparisons. An existing torch import alone does not establish extension compatibility. Later model tests pin model and tokenizer to `fe8a4ea1ffedaf415f4da2f062534de366a451e6`; keep model cache separate from exported results.

## Quota-aware execution

Default to smoke before the required profile. Smoke is a setup/correctness check, not publishable full-matrix performance evidence. Before each full case, check session remaining time where available and record memory headroom. End cleanly between cases when resources are insufficient. Never use keepalive automation, account sharing or coordinated quota circumvention as a prerequisite for reproduction.

Save artifacts after every completed case to the provider's supported output location and offer a manual download. On a fresh resumed session, assign a new attempt ID and parent reference. Re-run preflight and warmup; do not append incompatible timings to the old attempt. Follow the [artifact contract](../evaluation/02-artifact-contract.md).

## Troubleshooting

| Symptom | Investigation and resolution |
|---|---|
| `nvcc` missing | Record absent build toolkit; select a supported environment or install an explicitly compatible toolkit through provider-supported means |
| No kernel image / architecture error | Compare detected capability, build flags and wheel architectures; rebuild for qualified device |
| Undefined extension symbol | Compare torch/Python/compiler ABI with the build manifest; remove only this project's build cache and rebuild |
| Wrong CUDA stream result | Run T-K06 with events; inspect device guard and explicit launch stream rather than adding global synchronization |
| Numeric failure only at odd width | Inspect tail masks, warp zero-fill, dtype cast and alignment guards |
| OOM | Check prompt+generation budget, persistent model memory and active request count; lower declared cap and restart, labeling changed configuration |
| Optional compiler baseline fails | Record version/error as unavailable; do not replace it with eager and keep the compiled label |
| Stream never ends | Check worker exception propagation, stop flag, bounded handoff and terminal state; do not add arbitrary sleeps |
| Results differ across users | Compare GPU, driver, software, thermal/load state and protocol; rerun matched cases before attributing difference to the kernel |

Any cache deletion must target an explicitly identified project build/cache directory. Do not erase shared model caches or provider environments to repair an extension experiment.
