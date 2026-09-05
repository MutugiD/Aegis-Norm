# Compatibility and capacity matrix

Status: source-verified candidates, not a tested support claim. Reviewed 2026-09-05. The first GPU run is EXP-ENV. No GPU endpoint, driver report, or notebook environment has been supplied.

## F01 candidate C2

F01 uses Python 3.11-3.13, PyTorch 2.14.0/cu126 and CUDA toolkit 12.6 in an isolated Linux x86_64 T4 notebook environment. The [2.14 release](https://pytorch.org/blog/pytorch-2-14-release-blog/) lists cu126 distribution support. The direct 2.8.0 package audit found eight known vulnerabilities; 2.14.0 had none at review time. Full resolved-environment audits run in package CI. Neither audit establishes GPU compatibility. F01 preflight captures exact compiler, driver and memory; actual T4 build/run remains pending. See the [walkthrough](../implementation/01-foundation-walkthrough.md).

## Historical validation candidate C1 (superseded for F01)

| Component | Selected candidate / rule | Evidence and status |
|---|---|---|
| Host execution | Linux x86_64, Python 3.11 | Project default; notebook interpreter variation must be recorded |
| GPU | NVIDIA T4, capability 7.5, one device | [NVIDIA](https://www.nvidia.com/en-gb/data-center/tesla-t4/) and [Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html); actual allocation unverified |
| PyTorch | 2.8.0 CUDA 12.6 wheel | Official [wheel listing](https://pytorch.org/get-started/previous-versions/); existence verified, project execution untested |
| Build toolkit | CUDA toolkit 12.6; `TORCH_CUDA_ARCH_LIST=7.5` | Match candidate wheel/toolkit family; wheel installation alone does not provide a complete compiler toolchain |
| Host compiler | GCC 11.x candidate | Record exact GCC patch version; compile smoke is mandatory |
| Transformers | 4.56.2 | [Tagged Llama source](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/llama/modeling_llama.py) inspected; PyPI version metadata checked |
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Pin revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6` for model and tokenizer |
| Precision | FP16 model and activations; FP32 sums; FP32 operator testing | Explicit load override; config's stored BF16 is not a T4 default |
| Attention | Transformers `eager` first | Keeps required path independent of external attention packages; identical in both comparison arms |
| Triton / compile | Optional, not required | [Current upstream](https://github.com/triton-lang/triton) lists NVIDIA 8.0+; C1 T4 compilation may be unavailable |
| Other dependencies | Resolve during I1, export a complete lock/freeze with artifacts | No silent claim of a tested complete environment |

C1 is retained as the historical research candidate; use C2 for the initial build notebook. Do not downgrade provider drivers or silently uninstall preloaded packages. Use the notebook's dedicated environment. Changes to C2 require a recorded alternative candidate with the same correctness contract; later Transformers/model support must be requalified.

The driver's reported CUDA maximum, the wheel's runtime version, and `nvcc --version` are different facts. Record all three. Actual driver sufficiency is established by importing PyTorch, a CUDA tensor smoke operation, and compiling/running the extension; missing compiler support is not a correctness pass.

## Models and exclusions

| Target | First release status |
|---|---|
| Pinned TinyLlama with homogeneous FP16 Llama norms | Required integration candidate |
| Small randomly initialized Llama config, FP16 and FP32 | Required integration tests without weight download |
| Llama 8B / Mistral | Later model qualification; synthetic widths do not establish compatibility |
| 70B model | Context example only; not a single-T4 target |
| BF16, mixed input/weight dtype | Reference fallback only; no native BF16 release claim |
| CPU, unsupported model class | Original/reference behavior; never launch CUDA |
| P100, L4, other allocated GPU | Separate results category; do not label it T4 evidence |
| Windows native CUDA extension | Not part of required GPU release; local documentation editing is supported |

## Provider feasibility

Free Colab restricts remote-control access and primary use through an external web UI; availability and duration fluctuate. Required runs therefore stay notebook-driven. Do not depend on SSH, persistent API hosting, or quota-extension techniques. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Kaggle documentation provides notebook and usage guidance, but current pages did not expose full content through the research fetch. Earlier indexed guidance references variable quotas and more than one hardware generation. Treat assigned GPU, remaining quota, outbound access, and editor connection capability as session facts. No public-tunnel guarantee was verified. [Notebook documentation](https://www.kaggle.com/docs/notebooks), [usage guidance](https://www.kaggle.com/docs/efficient-gpu-usage).

## Capacity experiment

Use a maximum total model context of 2048 tokens, including chat-template tokens and generated tokens. Default demo generation cap is 128, maximum 512. Reject over-budget requests rather than silently truncating. Queue raw validated requests; tokenize on CPU, and allocate GPU inputs/KV only when active. Four waiting requests do not mean four concurrent GPU caches.

Measure peak allocated and reserved memory after warmup and during the longest supported request. Keep at least 1 GiB device headroom as a first-release admission target. If the environment cannot satisfy it, reduce the configurable context cap and report that configuration; never compare different caps as identical workloads.
