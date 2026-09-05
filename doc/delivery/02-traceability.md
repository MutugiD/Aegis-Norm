# Requirement, feature, test and evidence map

Status: release requirements specified; initial F01 reference/preflight and GPU smoke tests passed in the [contributor run](../evaluation/05-f01-gpu-evidence-review.md). Full native RMSNorm, integration and serving cases remain pending. Test definitions are in the [test plan](../testing/01-test-plan.md); evidence format is in the [evaluation protocol](../evaluation/01-protocol.md). Executed documentation-tool tests are recorded separately.

| Requirement | Feature | Tests | Required evidence |
|---|---|---|---|
| REQ-K01 | F01, F02 | T-K01, T-N01 | Build log, freeze, architecture and smoke output |
| REQ-K02 | F02 | T-K02, T-K03 | Per-case absolute/relative errors and classifications |
| REQ-K03 | F02 | T-K04, T-K05, T-K08 | Validation failures, empty/offset cases, sanitizer output |
| REQ-K04 | F02 | T-K06, T-K07 | Stream/device and mutation tests |
| REQ-K05 | F01, F02 | T-K01, T-K04 | Strict dispatch and fallback report |
| REQ-K06 | F03 | T-K09 | FakeTensor/opcheck output or explicit P1 deferral |
| REQ-K07 | F04 | T-K10 | Paired distributions and regression decision |
| REQ-I01 | F05 | T-I01, T-I02 | Parameter identities, patch paths and state comparisons |
| REQ-I02 | F05, F10 | T-I03, T-E03 | Save/reload results, logits and text comparisons |
| REQ-I03 | F05 | T-I04 | Training/unsupported/conflict outcomes |
| REQ-N01 | F06 | T-N01, T-N02 | Fresh-session transcript and explicit device identity |
| REQ-N02 | F06 | T-N03 | Complete or interrupted artifact bundle, checksums |
| REQ-N03 | F06 | T-N04 | Notebook-only run; separate optional tunnel status |
| REQ-S01 | F07 | T-S01, T-S02 | Schema/context/usage and real nonstream result |
| REQ-S02 | F08 | T-S06, T-S07 | Parsed SSE events, Unicode reconstruction, terminal outcome |
| REQ-S03 | F07, F08 | T-S03, T-S08 | Capacity, producer backpressure and slot lifecycle |
| REQ-S04 | F08 | T-S04, T-S07, T-S09 | Deterministic cancellation/deadline/failure injection |
| REQ-S05 | F07, F08 | T-S05, T-S10 | Readiness, authentication, shutdown and redacted logs |
| REQ-E01 | F09, F10, F11 | T-E01, T-E02, T-E04, T-E05 | Raw timings, metadata and comparable workload manifests |
| REQ-E02 | F09, F13 | T-E07 | Auditable report with unavailable/unmeasured fields |
| REQ-E03 | F12 | T-E06 | Separate contributor run IDs and reproduction assessment |

No test result is inherited from an upstream project or from a documentation check. A test marked skipped needs its reason and affected claim; a mandatory skipped GPU test blocks release qualification on that configuration. Documentation status can be complete while every row's execution evidence is pending.
