# Documentation index

Start with the [master plan](00-master-plan.md), then the [research findings](research/01-feasibility.md), [claim audit](research/02-context-audit.md), [compatibility matrix](research/03-compatibility.md), and [source register](research/04-sources.md).

The project is at the documentation milestone. No CUDA, model, or serving performance has been measured. Subsequent documentation PRs add architecture, PRDs, feature traceability, testing, evaluations, and operations in that order.

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

Detailed evaluation/artifact protocols, results status and operations guides follow in the next documentation increment.

[CI/CD](operations/04-ci-cd.md) validates documentation and repository tooling and delivers a documentation archive. It does not establish GPU correctness or performance.
