# Aegis-Norm master plan

Status: implementation specification; documentation milestone. Updated: 2026-09-05.

## Objective

Aegis-Norm is a small inference product with a custom CUDA RMSNorm operation. Explain and validate the complete request path: client, API, chat formatting, tokenization, prefill, model operations, fused normalization, cached decode, sampling, and response streaming. A faster isolated kernel is useful evidence, but does not establish a faster product.

The immediate deliverable is this research and documentation package. The CUDA package, notebooks, server, and GPU results described here are planned work, not existing functionality. Read the [documentation index](README.md) in order.

## Agreed boundaries

- Release 0.1: inference-only FP16/FP32 RMSNorm, reversible Llama integration, a pinned TinyLlama demonstration, independent T4 notebook runs, reproducible benchmarks, and a bounded streaming API.
- One active generation and four waiting requests in the first demo. No continuous batching in this release.
- Later product: browser interface, multi-request scheduling, supported model expansion, and operational capabilities selected using evidence.
- Contributors use their own allocated Colab/Kaggle sessions. Required work runs inside notebooks; tunnels are optional and provider-dependent.
- No mandatory paid hardware, GPU fleet, shared account, training/backward kernel, BF16 release requirement, or 70B model deployment.
- Report limited, negative, or inconclusive speedups honestly. Never fill results with illustrative numbers presented as measurements.

## Delivery phases

| Phase | Output | Exit condition |
|---|---|---|
| D0 | README-only repository bootstrap | Initial main pushed without rewriting remote history |
| D1 | Master plan, source register, feasibility, compatibility and claim audit | Claims distinguished from measurements; research-dependent design choices identified |
| D2 | Product definition, request lifecycle, architecture, interfaces, decisions, PRDs | First-release behavior is implementable; future proposals clearly labeled |
| D3 | Feature backlog, traceability, tests, evaluations, results templates, operations | Requirements mapped to tests and evidence; document links and examples checked |
| I1 | Build, reference, preflight | CPU reference and extension import contract work; T4 environment identified |
| I2 | Correct kernel | Numerical, layout, stream, and memory-safety tests pass on T4 |
| I3 | Optimizations | Each candidate compared against correct baseline; regressions recorded |
| I4 | Model integration | Patch, unpatch, generation, state preservation, reload verified |
| I5 | Notebooks and API | Fresh-session reproduction and bounded streaming functionality pass |
| E1 | Three-level evaluation | Raw artifacts, metadata, statistics, and limitations published |
| R1 | Release | Required tests pass; measured compatibility and known limits documented |

Research precedes architecture and PRDs. D1 findings inform D2; these documents do not claim a GPU feasibility experiment has occurred. I1 establishes the actual tested environment before I2. Optimizations may be rejected without blocking a correct reference release. E1 can demonstrate negligible product gains and still be a valid outcome.

## Repository workflow

Remote: `git@github.com:MutugiD/Aegis-Norm.git`. Initial branch: `main`. Initial README-only commit: `first commit`. Subsequent commits use `/task: ...` for research/documentation/setup and `/feat: ...` for product features. No generated author trailers or assistant branding.

Use one coherent prerequisite or feature per PR, with no more than one open PR at a time. Wait for green checks, merge, update main, then create the next branch from main. Do not stack PRs or force-push. The collected context is exploratory input, not authoritative requirements. This initial import preserves the original contents for history; the product-context increment curates it into Markdown and removes the text file.

## Completion and evidence

Documentation is complete when all first-release contracts, dependencies, numerical tolerances, failure behavior, test cases, and evaluation procedures are specified, with local links and structured examples validated. Every requirement has a feature and a test mapping. A future feature has a graduation gate rather than a false claim of implementation readiness.

The release additionally requires actual software and GPU evidence. Documentation checks cannot substitute for CUDA execution, functional testing, or model/serving measurements. Hardware-dependent unknowns have named experiments and remain unmeasured until artifacts exist.
