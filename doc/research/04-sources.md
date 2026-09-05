# Source register

Access date: 2026-09-05. Primary sources only. Moving pages describe capabilities at access time; their statements do not certify candidate C1. No third-party benchmark is an Aegis-Norm result.

| ID | Source and revision | Supports / limitation |
|---|---|---|
| S01 | [Zhang and Sennrich, RMSNorm, arXiv:1910.07467v1](https://arxiv.org/abs/1910.07467v1), 2019 | Algorithm and historical LayerNorm comparison; not custom-kernel speedup evidence |
| S02 | [NVIDIA T4 specifications](https://www.nvidia.com/en-gb/data-center/tesla-t4/), moving | Turing, memory capacity/type; vendor throughput is not achieved application bandwidth |
| S03 | [NVIDIA Turing tuning guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html), served as 13.3 | Capability, synchronization, occupancy and cache considerations |
| S04 | [PyTorch custom C++/CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), moving | Dispatcher integration, fake registration and opcheck; current stable-ABI guidance is not assumed available in C1 |
| S05 | [Transformers Llama implementation v4.56.2](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py) | Cast order, epsilon field, kernel hook; raw tagged source fetched after HTML rate limit |
| S06 | [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html), main | Asynchrony and stream/lifetime requirements |
| S07 | [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/), moving | 2.8.0 cu126 distribution exists; not proof of notebook compiler compatibility |
| S08 | [TinyLlama config](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/fe8a4ea1ffedaf415f4da2f062534de366a451e6/config.json), immutable model revision | Architecture, context, epsilon and stored dtype; revision independently retrieved via model API |
| S09 | [Transformers continuous batching](https://huggingface.co/docs/transformers/continuous_batching), moving | Existing scheduling capability; not a feature assumption for pinned v4.56.2 |
| S10 | [vLLM RMSNorm CUDA v0.6.6](https://github.com/vllm-project/vllm/blob/v0.6.6/csrc/layernorm_kernels.cu) | Prior fused implementation, dtype dispatch, guard and stream; historical source comparison |
| S11 | [Liger RMSNorm](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py), main | Prior Triton implementation and casting options; no T4 execution verified |
| S12 | [Triton compatibility](https://github.com/triton-lang/triton#compatibility), main | Current documented NVIDIA minimum; older versions require separate qualification |
| S13 | [Colab FAQ](https://research.google.com/colaboratory/faq.html), moving | Provider restrictions and variable allocation; recheck before any remote workflow |
| S14 | [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), moving | Provider documentation entry point; current body unavailable to fetch, no tunnel support conclusion |
| S15 | [Kaggle GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage), moving | Usage guidance entry point; inspect actual session for quota/hardware |
| S16 | [PyTorch RMSNorm API](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.normalization.RMSNorm.html), main | Explicit epsilon and normalized-shape semantics; actual C1 dispatch must be measured |
| S17 | [Transformers generation utilities](https://huggingface.co/docs/transformers/main/en/internal/generation_utils), main | Streamers and stopping interfaces; match implementation to pinned version |
| S18 | [Transformers 4.56.2 metadata](https://pypi.org/pypi/transformers/4.56.2/json) and [PyTorch 2.8.0 metadata](https://pypi.org/pypi/torch/2.8.0/json) | Release existence and Python metadata checked; transitive lock not yet tested |

## Evidence discipline

Cite a source beside externally supported findings. Label design choices as project decisions and calculations as estimates. Resolve moving references to release/commit identifiers when building an executable baseline, and export those identifiers with run artifacts. An inaccessible reference is recorded as unavailable rather than replaced with an assumed claim.

Local evidence consists of the preserved context and its static audit. The request-lifecycle material supplied during planning is explanatory input, not independent experimental evidence. Research was performed without accessing a GPU session.
