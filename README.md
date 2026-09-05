# Aegis-Norm

Aegis-Norm is an inference product initiative centered on a native CUDA RMSNorm operation, Hugging Face model integration, and a minimal streaming API. Its initial hardware target is a single NVIDIA T4 in an independent contributor-operated Colab or Kaggle session.

**Current stage: native RMSNorm passed initial T4 correctness testing.** The repository contains an installable Python package, the cast-ordered RMSNorm reference, environment preflight, a native vector smoke extension and an executable T4 build notebook. The complete F02 contributor run passed 223 native RMSNorm tests and 38 foundation tests, with compiler logs and verified hashes; see the [evidence review](doc/evaluation/06-f02-gpu-evidence-review.md). Native RMSNorm now has an explicit loader, dispatcher registration and a CUDA forward implementation. Sanitizer, independent reproduction and performance gates remain outstanding. Model integration and the inference server remain planned.

For native RMSNorm, use the [F02 walkthrough](doc/implementation/02-native-rmsnorm.md) and [correctness notebook](notebooks/02-t4-rmsnorm-correctness.ipynb). Start with the [F01 concept walkthrough](doc/implementation/01-foundation-walkthrough.md) and [T4 build notebook](notebooks/01-t4-build-smoke.ipynb). Compilation and required CUDA tests run inside the Colab/Kaggle session; no local CUDA installation is required. The native smoke calculates `y = 3 * x` to exercise the build/execution path; it is not RMSNorm.

## Planned first release

- Inference-only FP16/FP32 RMSNorm with explicit numerical and stream semantics.
- Reversible adaptation of qualified Llama normalization modules.
- Reproducible T4 testing and a pinned TinyLlama model demonstration.
- A chat-completions API with streaming, bounded admission, and cancellation.
- Separate kernel, model, and serving evaluations.

## Documentation

| Document | Purpose |
|---|---|
| [Master plan](doc/00-master-plan.md) | Scope, phases, delivery and completion criteria |
| [Documentation index](doc/README.md) | Complete reading map |
| [Product context](doc/product-context.md) | Curated exploratory ideas and request flow |
| [Research findings](doc/research/01-feasibility.md) | Evidence, alternatives and feasibility |
| [Feature breakdown](doc/delivery/01-feature-breakdown.md) | Incremental implementation and PR boundaries |
| [Test plan](doc/testing/01-test-plan.md) | Numerical, integration and API acceptance scenarios |
| [Evaluation status](doc/evaluation/03-results.md) | Current results and qualification status |

Repository CI validates documentation and Python code, builds and installs the package, tests reference semantics, audits dependencies, runs CodeQL, and produces documentation/package artifacts. CPU CI does not establish CUDA correctness or inference performance. GPU qualification requires notebook evidence.

## Evidence status

Kernel performance, model acceleration, serving throughput, and GPU memory effects are **not yet measured**. Environment choices are validation candidates, not tested support claims. The product context contains exploratory input; research and explicit design decisions determine the specification.

Continuous batching, additional models, a browser interface, and production hosting remain future capabilities.
