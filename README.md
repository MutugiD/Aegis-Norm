# Aegis-Norm

Aegis-Norm is an inference product initiative centered on a native CUDA RMSNorm operation, Hugging Face model integration, and a minimal streaming API. Its initial hardware target is a single NVIDIA T4 in an independent contributor-operated Colab or Kaggle session.

**Current stage: research, architecture, and implementation specification.** The repository contains product context, primary-source research, PRDs, feature dependencies, interface contracts, testing protocols, evaluation templates, and contributor guidance. The CUDA library, model integration, notebooks, and inference server are not implemented yet.

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

Repository CI validates documentation and Python tooling, audits dependencies, runs CodeQL, and produces a documentation artifact. These checks do not establish CUDA correctness or inference performance.

## Evidence status

Kernel performance, model acceleration, serving throughput, and GPU memory effects are **not yet measured**. Environment choices are validation candidates, not tested support claims. The product context contains exploratory input; research and explicit design decisions determine the specification.

Continuous batching, additional models, a browser interface, and production hosting remain future capabilities. No installation or runtime usage is advertised before working code exists.
