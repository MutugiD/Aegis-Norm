# PRD: RMSNorm library and model adaptation

Status: specified for v0.1. Audience: kernel/inference engineers. Problem: a model-specific eager normalization path may spend avoidable time in launches and intermediate materialization. Outcome: a correct replaceable operator whose effects can be measured in isolation and in a model.

## Requirements

| ID | Priority | Requirement and acceptance |
|---|---|---|
| REQ-K01 | P0 | Source-build a registered operator for the C1 T4 candidate; CPU package import/reference must not require CUDA compilation |
| REQ-K02 | P0 | FP16/FP32 native output follows the declared casting/epsilon contract and passes numerical tolerances |
| REQ-K03 | P0 | Validate shape, dtype, device, epsilon, gradient use and limits; cover empty rows and scalar-safe offsets without invalid memory access |
| REQ-K04 | P0 | Use the correct device/current stream; do not mutate inputs or introduce device-wide synchronization |
| REQ-K05 | P0 | Expose auto/cuda/reference modes with inspectable dispatch; strict native mode never silently falls back |
| REQ-K06 | P1 | Add FakeTensor registration and opcheck coverage; full-model compile remains an optional experiment |
| REQ-K07 | P0 | Optimize only with paired numerical/performance evidence across decode and prefill cases; retain correct baseline when candidate loses |
| REQ-I01 | P0 | Adapt only qualified Llama instances transactionally; preserve parameter identity/state keys; patch/restore are idempotent |
| REQ-I02 | P0 | Generate, save, reload and restore correctly; verify teacher-forced logits and generated outputs separately |
| REQ-I03 | P0 | Preserve original training/unsupported behavior in auto mode; strict mode reports incompatibility without global class changes |

The [operator contract](../architecture/02-operator-contract.md) defines exact inputs, errors and adaptation behavior. Priority P0 means release gate; P1 is a desired capability that may be explicitly deferred without pretending coverage. No backward kernel, arbitrary model-family replacement, BF16 native path, quantization coupling or residual-add fusion in v0.1.

## User flow and acceptance

Engineer installs the future source package, runs preflight, executes reference/native on the same input, patches a loaded model, observes the patched count, runs generation, restores and verifies parameter identity. The model still loads normally from saved weights. If the extension cannot load, reference import works and strict native use fails clearly.

Release acceptance requires on-device numerical and stream tests, model integration checks and recorded benchmark outcomes. A scalar baseline may ship when packed/vectorized candidates are slower; an optimization claim requires passing evidence. Stable interfaces are limited to the documented Python API and registered schema; native C++ symbols are internal.
