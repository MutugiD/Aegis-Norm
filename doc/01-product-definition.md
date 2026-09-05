# Product definition

Status: specified, 2026-09-05. [Master plan](00-master-plan.md) controls scope; [research](research/01-feasibility.md) controls factual claims.

## Users and outcomes

| User | Job | First-release outcome |
|---|---|---|
| Kernel/inference engineer | Replace and measure a model operation | Explicit native operator, reversible adaptation, three-level evidence |
| GPU contributor | Reproduce a result on their allocated T4 | Notebook setup, preflight, bounded runs and downloadable artifacts |
| Demo user | Send a chat request and read the response | Validated HTTP request, streaming or complete response, understandable failures |
| Maintainer/reviewer | Assess each change independently | Small PRs, requirement/test/evidence traceability, reproducible limitations |

The product thesis is that reducing normalization overhead can improve some inference workloads while preserving model behavior. This thesis is tested rather than promised. The initial product serves one configured model; users do not select arbitrary downloads or inspect kernel controls in the chat flow.

## First-release use cases

1. An engineer executes reference and native RMSNorm on identical tensors, checks numerical equivalence, and compares repeated timings.
2. The engineer loads the pinned model, adapts supported norms, generates a response, then restores original behavior without changing weights.
3. A contributor runs the notebook from a fresh session and exports complete metadata/results even when an optional baseline is unavailable.
4. A client sends one chat completion request, receives correctly framed output, or cancels without leaving an orphaned generation.
5. A reviewer distinguishes isolated kernel improvement from model and client-observed effects.

## Product boundaries

The API, generation worker, model runtime and native operation are separate responsibilities. First release uses a single model process, one active generation and four waiting requests. This is serialized serving with admission control, not continuous batching. Client UI, multi-tenancy, accounts, billing, distributed execution, paged-cache implementation, backward kernels, and wider model support are future work.

## Outcome measures

Required: all mandatory functionality/correctness tests pass; a fresh T4 environment can build and run; raw benchmark artifacts are complete; unsupported configurations are identified. Performance: report medians, dispersion and paired speedup intervals, including regressions. No universal percentage target is set. A scientifically useful no-speedup result can meet the engineering milestone while changing the later product roadmap.

## Terminology

| Term | Project meaning |
|---|---|
| RMSNorm | Per-token last-dimension normalization without mean subtraction |
| Gamma | Existing learned scale vector; not changed by adaptation |
| Prefill | Initial processing of prompt tokens and construction of K/V state |
| Decode | Subsequent model steps using new input tokens and retained K/V state |
| TTFT | Time to first generated token at model level, or first nonempty text delta at client level; always label which |
| ITL | Model generated-token interval; SSE text-chunk intervals are reported separately |
| TPS | Generated tokens divided by a stated measurement interval, excluding prompt tokens |
| Global memory | Device tensor storage; GDDR6 on T4 |
| Queueing | Waiting for the single worker; it does not combine model work |
| Continuous batching | Runtime rescheduling of active requests across generation steps; future feature |
| Fallback | Explicit execution of original/reference math for unsupported native cases |

The original title's enterprise ambition describes direction, not a reliability, security, or scale certification.
