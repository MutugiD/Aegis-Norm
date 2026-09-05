# Documentation index

Start with the [master plan](00-master-plan.md), then the [research findings](research/01-feasibility.md), [claim audit](research/02-context-audit.md), [compatibility matrix](research/03-compatibility.md), and [source register](research/04-sources.md).

The documentation milestone is complete and F01 foundation code is implemented, with passing foundation and native RMSNorm tests in the complete F02 archive. No CUDA, model, or serving performance has been measured. The [F01 walkthrough](implementation/01-foundation-walkthrough.md) explains the foundation. The [F02 walkthrough](implementation/02-native-rmsnorm.md) covers the new native RMSNorm implementation and qualification notebook; its initial 223-case GPU suite passed, as recorded in the [F02 evidence review](evaluation/06-f02-gpu-evidence-review.md).

Status vocabulary: **specified** means intended behavior; **source-verified** means checked against an identified source; **candidate** means selected for experiment; **tested** requires a recorded execution; **not yet measured** means no performance evidence exists.

Collected ideas are curated in [product-context.md](product-context.md), with clear sections, readable diagrams and research questions at the bottom. They are exploratory input, not an authoritative specification. The original text is preserved in Git history and its superseded file is removed.

## Product specification

- [Product definition](01-product-definition.md) and [design exploration](research/05-design-exploration.md).
- [System and request lifecycle](architecture/01-system-and-lifecycle.md).
- [Operator and integration contract](architecture/02-operator-contract.md).
- [HTTP API and runtime contract](architecture/03-api-and-runtime.md).
- [Architecture decisions](architecture/04-decisions.md).
- PRDs: [kernel library](prd/01-kernel-library.md), [testing/demo](prd/02-testing-and-demo.md), [future product](prd/03-future-serving-product.md).
- [Feature breakdown](delivery/01-feature-breakdown.md), [traceability](delivery/02-traceability.md), and [test plan](testing/01-test-plan.md).

## Evaluation and operations

- [Evaluation protocol](evaluation/01-protocol.md): kernel, model and serving measurements.
- [Artifact contract](evaluation/02-artifact-contract.md), [regression fixtures](evaluation/fixtures.json), and [not-run manifest](evaluation/manifest.example.json).
- [Results and qualification status](evaluation/03-results.md).
- [Contributor guide](operations/01-contributor-guide.md), [deployment and remote access](operations/02-deployment-and-access.md), and [risk/release checklist](operations/03-risk-and-release.md).
- [Documentation validation record](delivery/03-documentation-validation.md).

[CI/CD](operations/04-ci-cd.md) validates documentation and repository tooling and delivers a documentation archive. It does not establish GPU correctness or performance.
