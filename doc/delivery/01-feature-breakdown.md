# End-to-end feature and PR breakdown

Status: implementation backlog, not completed software. [Traceability](02-traceability.md) maps requirements to tests. D0-D3 are documentation increments; F01-F13 are subsequent implementation/evaluation increments.

F01 code and its initial notebook are implemented; the contributor run passed both GPU smoke tests. Standalone build metadata and a qualified environment lock remain tracked evidence gaps, not a claim that GPU execution failed. See the [concept and code walkthrough](../implementation/01-foundation-walkthrough.md). Supplementary CPU checks do not complete this feature's GPU acceptance gate.

| Feature | Depends on | Deliverable / PR boundary | Acceptance and evidence |
|---|---|---|---|
| D0 Bootstrap | Empty remote check | README-only main and remote | Initial commit pushed without rewriting history |
| D1 Research foundation | D0 | Master plan, source register, feasibility, context preservation | Primary-source claims and experiment gates recorded |
| D2 Architecture and PRDs | D1 | Lifecycle, operator/API contracts, decisions, product requirements | Complete first-release behavior; future scope separated |
| D3 Validation and delivery | D2 | This backlog, tests, evaluations, results templates, operations | Links/examples/traceability checked; no invented results |
| F01 Package and preflight | D3 | Python package/reference, source-build metadata, environment report, minimal T4 setup/build notebook | Compile and execute a native smoke operation in a C1 or documented C2 T4 notebook; run reference on CUDA tensors; export build/run logs and resolved dependency lock |
| F02 Correct native RMSNorm | F01 | Dispatcher, C++ validation, scalar CUDA reduction and strict native API | T-K01..T-K08 pass on T4; preserve FP16 cast boundary |
| F03 Framework registration | F02 | FakeTensor support, opcheck and unsupported-autograd behavior | T-K09; no full-model compilation claim |
| F04 Kernel experiments | F02 | One measured candidate per PR: geometry, packing, then optional value retention | T-K10 plus full correctness regression; reject losing candidates rather than merging for novelty |
| F05 Model adaptation | F02 | Transactional instance patch, report, restore and original-forward fallback | T-I01..T-I04 with random small Llama and pinned TinyLlama |
| F06 Contributor notebooks | F01, F02, F05 | Colab and Kaggle entry notebooks, bounded profiles, export | T-N01..T-N04; absent optional providers/baselines explicitly skipped |
| F07 Nonstream generation API | F05 | Schemas, one worker, admission, readiness, nonstream response | T-S01..T-S05; one real model request plus fake-worker failures |
| F08 Streaming and lifecycle | F07 | Bounded text handoff, SSE, cancellation/deadlines/shutdown | T-S06..T-S10; no orphan worker or event-loop blocking |
| F09 Kernel benchmark harness | F02 | Shape matrix, CUDA-event/host timing, baseline equivalence, raw sample export | T-E01..T-E02; EXP-KERNEL completed with strict dispatch |
| F10 Model evaluation harness | F05, F09 | Quality fixtures, prefill/cached decode, memory, profile attribution | T-E03..T-E04; EXP-MODEL artifacts and conclusions |
| F11 Serving evaluation harness | F08, F10 | Fixed-workload local client, concurrency and optional tunnel classification | T-E05; EXP-SERVE including rejected/failed requests |
| F12 Independent reproduction | F06, F09, F10, F11 | Contributor run review, matched-environment comparison | T-E06; report repeatability without pooling unlike hardware |
| F13 Release evidence | F03, F04, F08, F12 | Result report, support matrix, limitations and release notes | T-E07 and release checklist; F03 deferral must be explicit if P1 omitted |

F04 can conclude that the scalar baseline remains preferable. Its experiments may use the provisional F09 harness; the final published comparison must follow the evaluation protocol. F03 is P1; F13 accepts a documented deferral rather than requiring a feature falsely declared complete. All P0 requirements remain mandatory.

## Per-feature implementation handoff

Each PR body states problem, resulting behavior, affected requirement IDs, exact tests run, environment/run IDs, and limits. Numerical behavior changes require a contract/ADR update before new results are compared. Source optimizations retain the previous correct kernel as a benchmark baseline, even if not exposed to end users.

Initial planned package surfaces are `aegis_norm.ops`, `aegis_norm.integrations`, C++/CUDA `csrc`, `server`, `tests`, `benchmarks` and `notebooks`. F01 creates executable commands and documents them; commands in these planning documents are not advertised as currently runnable modules.

F01 targets compilation and execution inside the contributor's Colab/Kaggle Linux T4 session. The notebook VM's CPU runs the host compiler and `nvcc`; its T4 executes the resulting CUDA code. F01 includes a minimal native smoke operation to verify this path before F02 implements RMSNorm. The smoke operation does not establish RMSNorm correctness or speed. A usable setup/build notebook is required in F01; F06 expands this into the full contributor workflow, model runs and artifact recovery. No local CUDA build is required. CPU-only package import and reference checks are supplementary CI checks, not the feature's acceptance evidence.

## Learning progression

Optional CUDA exercises run in the GPU notebook and progress through scalar multiplication, vector addition, block reduction, warp reduction, then RMSNorm. They are learning aids, not product features or substitutes for tests. Keep exercises outside release benchmarks so product evidence always measures the declared operator and model.

## Later-product backlog

FUT-01 client UI follows F08; FUT-02 batching follows F11; FUT-03 models follows F10; FUT-04 tenant operations follows an explicit production hosting decision; FUT-05 residual fusion follows measured attribution; FUT-06 distribution/hosting follows F13. Graduation criteria are in the [future PRD](../prd/03-future-serving-product.md).
