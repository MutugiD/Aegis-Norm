# Documentation index

Start with the [master plan](00-master-plan.md), then the [research findings](research/01-feasibility.md), [claim audit](research/02-context-audit.md), [compatibility matrix](research/03-compatibility.md), and [source register](research/04-sources.md).

The project is at the documentation milestone. No CUDA, model, or serving performance has been measured.

## Product and architecture

1. [Product definition and terminology](01-product-definition.md).
2. [System architecture and request lifecycle](architecture/01-system-and-lifecycle.md).
3. [Normalization and integration contracts](architecture/02-operator-contract.md).
4. [HTTP API and runtime contract](architecture/03-api-and-runtime.md).
5. [Architecture decisions](architecture/04-decisions.md).

## Product requirements

- [Kernel library and model adaptation](prd/01-kernel-library.md).
- [Contributor testing and streaming demo](prd/02-testing-and-demo.md).
- [Future serving product](prd/03-future-serving-product.md), proposed and outside v0.1.

Feature traceability, testing, evaluations, and operations follow in the next documentation increment.

Status vocabulary: **specified** means intended behavior; **source-verified** means checked against an identified source; **candidate** means selected for experiment; **tested** requires a recorded execution; **not yet measured** means no performance evidence exists.

Original input: [context.txt](../context.txt). The additional request-lifecycle explanation supplied during planning informs the architecture; corrections are recorded in the claim audit.
