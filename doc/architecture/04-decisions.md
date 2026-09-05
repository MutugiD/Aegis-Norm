# Architecture decision records

Status: selected for implementation, pending identified GPU experiments. These are decisions, not claims of achieved compatibility.

| ID | Decision | Alternatives and rationale | Revisit when |
|---|---|---|---|
| ADR-001 | Native C++/CUDA, dispatcher registration, Python wrapper | Plain pybind alone has weaker framework integration; current Triton support does not establish T4 viability. Source-built C1 keeps experiment scope explicit | A tested alternative wins on correctness, portability and performance |
| ADR-002 | Homogeneous FP16/FP32 native path with Llama cast boundary | Single final cast changes semantics; BF16 and arbitrary mixed dtypes expand validation beyond T4 MVP | Separate precision PR has qualifying hardware and tests |
| ADR-003 | One row/block, 256 threads, scalar-safe first | Register/shared-row caching and packed loads may improve traffic but add occupancy/alignment risks | EXP-KERNEL supplies paired evidence for shape-specific dispatch |
| ADR-004 | Patch existing module instances and preserve original forward | Global class monkey-patching affects unrelated models; replacement modules complicate identity/state preservation; hub hooks add distribution dependencies | A versioned hook adapter passes the same state/restore tests |
| ADR-005 | One generation worker, four waiting requests | Continuous batching has more cache/scheduling complexity than needed to prove the vertical slice | Future scheduler PR evaluates supported Transformers APIs and other runtimes on actual hardware |
| ADR-006 | Eager attention and pinned TinyLlama for C1 | New attention packages or quantization can hide normalization effects and complicate T4 support | A separately labeled model/attention qualification is complete |
| ADR-007 | Notebook-driven required workflow, optional HTTP tunnel | Free sessions are ephemeral and remote access may be restricted | Provider-supported remote workflow is demonstrated and documented |
| ADR-008 | Three-level evaluation, no unconditional performance target | Kernel-only results cannot establish model/client value; fixed large speedup targets bias reporting | Product requirements change using measured evidence |
| ADR-009 | Explicit reference fallback, strict native benchmark mode | Silent fallback can make benchmarks falsely appear to measure native code | New backend reports dispatch reliably and passes tests |
| ADR-010 | Initial source build, no stable-ABI promise | Candidate PyTorch 2.8 predates newer stable-ABI guidance; prebuilt wheels multiply compatibility work | Build matrix and release maintenance justify binary distribution |

Evidence: [feasibility](../research/01-feasibility.md), [compatibility](../research/03-compatibility.md), [source register](../research/04-sources.md). Any implementation that changes these decisions updates this document, affected PRDs, tests and result comparison rules in the same PR.
