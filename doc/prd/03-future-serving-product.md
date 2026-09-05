# PRD: future serving product

Status: proposed direction, outside v0.1. This document is intentionally a set of gated capabilities, not a promise that unresearched enterprise requirements are settled.

## Product opportunities

| ID | Capability | User outcome | Graduation evidence |
|---|---|---|---|
| FUT-01 | Browser chat client and cancellation | A nontechnical tester can use the validated API | API v0.1 released; client usability and disconnect tests |
| FUT-02 | Multi-request continuous batching | More useful work per GPU iteration at acceptable latency | Compare existing supported Transformers/runtime batching with serialized baseline; measure fairness, cancellation and memory |
| FUT-03 | Additional models and precision modes | Apply optimization to more useful workloads | Exact model/version/hardware qualification, not similarity of class names |
| FUT-04 | Authentication, tenant quotas and operational metrics | Safely operate a maintained shared endpoint | Defined tenant model, supported host, threat model and load/SLO evidence |
| FUT-05 | Residual-add normalization fusion | Remove another materialized boundary when appropriate | Profiled opportunity, alias/mutation contract and end-to-end benefit |
| FUT-06 | Packaged releases and supported hosting | Repeatable deployment beyond ephemeral notebooks | Maintained dependency matrix, capacity plan, rollout/rollback and cost measurements |

Prioritize FUT-01 after the demo if usability is the blocker, FUT-02 if measured queue delay dominates, or FUT-05 if model profiling identifies an operation boundary worth fusing. Do not implement all capabilities simply because they appear in the roadmap.

## Runtime selection gate

Before scheduling work, evaluate versioned Transformers continuous batching and at least one suitable existing inference runtime. Verify T4 and kernel-extension integration, cache management, cancellation, memory budgets, throughput and tail latency. Reuse a qualifying runtime rather than presupposing a handwritten scheduler. If none qualifies, write a separate scheduling architecture and acceptance specification before implementation.

## Deliberately unresolved product decisions

Production hosting budget, tenant model, reliability SLO, supported model catalog and deployment scale are not inferable from free notebook experiments. Resolve them with measured v0.1 evidence and product requirements before graduating FUT-04/FUT-06. No uptime, concurrency, security certification or cost-per-request claim is made here.
