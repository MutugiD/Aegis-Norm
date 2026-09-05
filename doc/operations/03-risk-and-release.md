# Risks, qualification and release checklist

Status: documentation milestone. Maintainer owns release decisions; contributors supply reproducible evidence. All GPU-dependent risks remain open until their named checks run.

| Risk | Consequence | Mitigation / evidence gate |
|---|---|---|
| Incorrect reduction/casts | Silent model corruption | T-K02/T-K03 before benchmarks; T-E03 model regression |
| Wrong stream/device/lifetime | Races or intermittent failures | T-K06/T-K07 and sanitizer qualification |
| Input retention spills registers | Optimization becomes slower | Compiler diagnostics plus F04 paired results |
| Existing implementation is equally fast | Weak product differentiation | Publish comparisons; prioritize integration/reproduction or adopt better backend |
| Tiny decode workload dominated by launch/host cost | Kernel improvements vanish | Separate operator/device/model timings and profile attribution |
| Version or provider environment drift | Build/import failure | C1/C2 manifests, exact locks, fresh-session preflight |
| Global or conflicting patches | Unrelated model behavior changes | Instance scope, transactionality, restore and conflict tests |
| Slow client / ignored cancellation | Memory growth or orphan work | Bounded handoff, stop grace, readiness failure rather than slot reuse |
| Short free sessions | Missing results | Case-level export, interrupted manifests, independent attempts |
| Quota/tunnel restrictions | Unusable deployment path | Notebook-only required workflow; optional access qualification |
| Noisy free hardware | Misleading speedup | Paired ordering, repeated trials, cross-session metadata |
| Limited regression corpus | Overclaimed quality preservation | Label corpus scope; broader evaluation before model/product expansion |
| Historical dependency candidate | Maintenance/public-host risk | Keep C1 experimental; qualify updated supported stack before public production |
| Repository integration permissions differ | Some publication tools may fail | Use the authorized repository credential; keep it out of files and CI; verify the resulting PR/check state |

## Documentation gate D3

- Master plan and index precede research; exploratory product context is clearly labeled.
- First-release interfaces and failure behavior have no unspecified implementation choices.
- Requirements map to features/tests, and each feature has a dependency and PR boundary.
- Local Markdown links, JSON examples, source references and context preservation are checked.
- Structured artifact examples are `not_run`; no software/GPU passes are claimed.
- Future product decisions are explicitly gated rather than quietly included in v0.1.

## Software release gate R1

- Candidate environment becomes a tested row only after fresh T4 build, import, numerical and stream checks.
- All P0 requirements pass; any P1 deferral is explicit in release notes.
- Memory safety/synchronization qualification is recorded; lack of sanitizer tooling remains a pending gate.
- Patch/restore and model save/reload preserve parameters; quality protocol and real request path pass.
- API functionality includes overload, cancellation, timeout, slow-client and shutdown scenarios.
- Kernel/model/serving artifacts are complete, checksummed and independently reviewed; reproduction status is accurate.
- Release notes report affected shapes, regressions, supported configurations, unsupported features and reproducibility limits.
- No credentials/private prompts/build binaries accidentally included; source/third-party license decisions resolved before distributing code.

## Rollout and monitoring

First ship a tagged experimental source release with a tested environment and manual contributor notebook. Warm and validate native behavior before readiness. Inspect failures, queue delay, timeouts, per-request generated count, first token/text times, and memory. A request failure is not automatically retried behind the user's back.

Rollback to a separately started reference configuration when native qualification fails. Preserve failing artifacts and document the regression; do not replace a failed run with an unrelated successful sample. Re-run affected gates after kernel/compiler/library changes. Broaden testing only when changes, failures or new support claims justify it.

## Unresolved external inputs

An actual GPU session is needed for EXP-ENV and onward; a second contributor is needed for independent reproduction. A maintained production host and explicit product targets are needed for future shared serving. These are evidence/resource dependencies, not omissions in the first-release contract.
