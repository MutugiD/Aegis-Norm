# Contributor and notebook workflow

Status: implementation runbook. The product package, test commands and notebooks are not yet implemented. This guide specifies what F01/F06 must make reproducible; the current repository contains documentation.

## Local development and review

Use the repository checkout for editing and CPU tests after their implementation. GPU execution targets Linux notebook sessions; Windows native compilation is not a v0.1 requirement. Read the [compatibility candidate](../research/03-compatibility.md), [operator contract](../architecture/02-operator-contract.md) and [test plan](../testing/01-test-plan.md) before native changes.

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

The C1 initial wheel selection is `torch==2.8.0` from the official cu126 index and `transformers==4.56.2`. A complete dependency lock is produced in F01 after resolution and smoke verification. Match toolkit family and record GCC; do not mistake `nvidia-smi`'s CUDA display for installed nvcc. Source builds use `TORCH_CUDA_ARCH_LIST=7.5` for the required T4 artifact and conservatively set `MAX_JOBS=2` to limit host build memory.

Do not automatically replace a notebook's Python/driver/toolkit to force C1. If C1 cannot run, capture the provider environment as C2 and review its compatibility before comparing it with C1. An existing torch import alone does not establish extension compatibility. Pin model and tokenizer to `fe8a4ea1ffedaf415f4da2f062534de366a451e6`; keep model cache separate from exported results.

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
