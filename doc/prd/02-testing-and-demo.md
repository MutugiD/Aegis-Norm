# PRD: contributor testing and streaming demo

Status: specified for v0.1. Audience: contributors, demo clients and reviewers. The demo proves that the normalization operation participates in a real inference request; it does not certify production hosting on a free notebook.

## Requirements

| ID | Priority | Requirement and acceptance |
|---|---|---|
| REQ-N01 | P0 | A fresh notebook identifies hardware/toolchain, installs the selected environment, builds and runs smoke tests; unsupported allocation is explicit |
| REQ-N02 | P0 | Export metadata and partial/complete artifacts without secrets; preserve failures and interrupted status |
| REQ-N03 | P0 | Core testing runs entirely inside a contributor notebook; tunnels are optional and separately qualified |
| REQ-S01 | P0 | Validate the defined chat-completions subset and context budget; produce correct nonstream response and token usage |
| REQ-S02 | P0 | Produce valid SSE role/content/finish/terminal framing; preserve Unicode; distinguish post-header errors |
| REQ-S03 | P0 | Enforce one active request, four waiting slots, bounded preprocessing and text handoff; overload is explicit |
| REQ-S04 | P0 | Handle disconnect, cancellation, deadlines and worker exceptions without orphaned work or premature slot release |
| REQ-S05 | P0 | Warm up before readiness; separate health/readiness; protect optional public access; omit prompts/secrets from normal logs |
| REQ-E01 | P0 | Record comparable kernel/model/serving evidence with environment and dispatch metadata |
| REQ-E02 | P0 | Label unavailable, failed, interrupted and unmeasured results distinctly; no fabricated performance numbers |
| REQ-E03 | P0 | Complete an independent contributor reproduction before labeling results independently reproduced |

Defaults and error codes are authoritative in the [HTTP contract](../architecture/03-api-and-runtime.md). Client sends messages for `tinyllama`, waits for admission, consumes content or a terminal error, and can disconnect. Contributor runs correctness before timing and exports artifacts for review.

## Acceptance scenarios

One ordinary request completes in streaming and nonstreaming forms. Five admitted requests occupy one active slot and four waiting slots; the next arrival is rejected. A queued cancellation frees capacity. A disconnected active request stops cooperatively before the next starts. A stalled worker makes readiness false. Unknown fields and overlong templated prompts are rejected. First-content timing ignores role events and heartbeats.

All of these require automated functionality tests, with a fake generation worker for deterministic error injection and a real T4 model for the successful vertical path. GPU results remain pending until executed.
