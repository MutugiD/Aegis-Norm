# Research and feasibility findings

Status: desk research completed 2026-09-05; GPU experiments not yet run. [Sources](04-sources.md) distinguish pinned code from moving documentation. These recommendations are engineering judgments, not benchmark results.

## Finding

A T4-focused inference RMSNorm extension is a feasible engineering candidate. Its value proposition is reduced intermediate materialization and launch overhead with explicit framework semantics. It is not a novel normalization algorithm or a replacement for a complete serving runtime. Existing fused implementations make comparison essential.

The original RMSNorm paper establishes normalization without mean subtraction; its reported comparisons concern RMSNorm versus LayerNorm in its experiments, not Aegis-Norm versus current inference kernels. Do not import those speedups into this project. [S01: paper](https://arxiv.org/abs/1910.07467).

## Where improvement could come from

The mathematical operation is `y_i = x_i * rsqrt(mean(x*x) + eps) * gamma_i`. The selected Llama reference adds observable casting stages: accumulate and normalize in FP32, cast normalized values back to input dtype, then multiply by the weight. A fused implementation must preserve these semantics within declared tolerances. [S05: Llama v4.56.2](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py).

Eliminating intermediate tensors can reduce launches and global-memory traffic. Actual improvement depends on row count, width, cache residency, allocations, CPU dispatch, and the baseline implementation. At one decode row, one-block-per-row exposes only one block: low total work and launch overhead may dominate. Large prefill matrices offer more block parallelism but are not the same workload. One default geometry is therefore a correctness starting point, not a universal optimum.

For a measured RMSNorm time fraction `f` and kernel speedup `s`, the optimistic isolated substitution bound is `1 / ((1-f) + f/s)`. A hypothetical `f=0.05, s=2` gives approximately `1.0256x`; this is an algebraic illustration, not a project result. Changed dispatch or allocation costs can further alter model behavior. Measure f separately with profiling, then time unprofiled execution.

## Alternatives and baseline decision

| Alternative | Research observation | Aegis-Norm treatment |
|---|---|---|
| Eager Llama RMSNorm | Clear model-specific casting and epsilon behavior | Required semantic and performance baseline, fixed revision |
| Native PyTorch RMSNorm | Public operator with explicit epsilon option | Required attempted comparison; check cast differences before calling it equivalent |
| Compiled expression | Compiler fusion may remove eager overhead | Attempt on compatible environments; unavailable backend is a recorded skip, not a loss |
| vLLM RMSNorm | Pinned CUDA source demonstrates existing fused normalization and current-stream/device integration | Source comparator; optional executable baseline only when its full stack supports the chosen environment |
| Liger RMSNorm | Existing Triton implementation offers model-related casting modes | Research comparator; do not force incompatible Triton into the required T4 environment |
| Hugging Face kernel hook | Present on the pinned Llama class | Evaluate later as distribution integration; first release controls adaptation per model instance |

Sources: [native RMSNorm](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.normalization.RMSNorm.html), [vLLM v0.6.6](https://github.com/vllm-project/vllm/blob/v0.6.6/csrc/layernorm_kernels.cu), [Liger source](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py). The vLLM tag is an inspected historical reference, not a recommended current serving deployment. Preserve applicable third-party notices if code is later reused; current documents specify an independently written kernel.

## Selected approach

Use a C++/CUDA extension registered with the PyTorch dispatcher, exposed through a small Python API. Add FakeTensor registration and `opcheck` coverage for the contract, without promising full-model compilation. PyTorch documents this custom-operator integration path. [S04](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).

Use one block per row, 256 threads, FP32 accumulation, full-warp reductions with zero-filled inactive work, and a scalar-safe path first. Optimize only after correctness. Candidate tuning compares 128/256/512 threads, packed loads, and value retention. Each candidate must justify itself against small decode and larger prefill workloads. Do not promise one DRAM read, perfect bandwidth utilization, or a fixed percentage instruction reduction.

Adapt only explicit supported Llama module instances after loading the model. Retain original methods and parameters so unpatch is reversible. Model execution stays on one GPU. No training, quantized-weight integration, residual-add fusion, or framework-wide monkey patch in v0.1.

## Product and infrastructure feasibility

TinyLlama is selected to make a full model run practical, not to simulate a 70B deployment. Its pinned config has 22 layers, width 2048, 32 attention heads and four KV heads, a 2048-token context, and epsilon 1e-5. Force FP16 despite the stored BF16 metadata. [S08](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json).

Approximate model-weight storage using the 1.1B label is 2.2 billion bytes in FP16, before allocator, activations, caches, and runtime overhead. Derived KV storage per token is `2 * 22 * 4 * 64 * 2 = 22528 bytes`, or 44 MiB for one 2048-token sequence. This is a capacity estimate, not a measured peak. Model offload and quantization are excluded from comparable first-release timings.

The demo uses a single generation worker and bounded queue. Current Transformers has documented continuous batching APIs; evaluate their version and backend requirements in the later scheduling feature rather than rebuilding scheduling by assumption. [S09](https://huggingface.co/docs/transformers/continuous_batching).

## Experiments needed before release

| Experiment | Question | Required disposition |
|---|---|---|
| EXP-ENV | Can the candidate build/load/run on the allocated T4? | Publish exact environment and smoke result; revise candidate if incompatible |
| EXP-NUM | Does the fused operator preserve reference behavior? | Pass predeclared FP16/FP32 tolerances and special-value classifications |
| EXP-KERNEL | Which shape families benefit or regress? | Paired latency distributions, including scalar-safe cases |
| EXP-MODEL | Does isolated kernel improvement survive integration? | Teacher-forced comparison, prefill/decode timing and measured RMSNorm share |
| EXP-SERVE | Does client experience improve without hiding queue delay? | Local and optional tunnel runs reported separately |
| EXP-REPRO | Can another contributor reproduce setup and conclusions? | Independent environment/artifacts, no pooled incompatible timings |

Recommendation: proceed with the specified documentation and correctness-first implementation. Product acceleration remains an open experimental question.
