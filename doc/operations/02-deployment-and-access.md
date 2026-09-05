# Deployment, remote access and operational limits

Status: first-demo specification; no deployed endpoint. The free GPU sessions are experimental execution environments, not a reliable production host.

## Supported first path

Run model, tests, benchmark client and optional loopback server inside the notebook session. One model process owns one T4. The local workstation edits/reviews documents and receives exported artifacts. This path needs neither remote shell nor public tunnel and remains the release's required reproduction workflow.

## Remote access matrix

| Workflow | Colab free | Kaggle | Other explicitly authorized host |
|---|---|---|---|
| Interactive notebook testing | Required candidate | Required candidate | Optional |
| Remote development / SSH | Not a supported free-Colab path under current FAQ restrictions | Not verified; use provider-supported editor integration only if available | Qualify host access and environment before use |
| Public HTTP demonstration | Not a dependency; check current provider restrictions before any use | Not verified; session and provider support must be established | Allowed when operator controls host/network and configures access |
| Persistent production service | Outside scope | Outside scope | Later hosting PRD and operations qualification |

Sources: [Colab restrictions](https://research.google.com/colaboratory/faq.html), [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks). Do not interpret a tunnel tool's technical ability to connect as provider approval. No provider-specific tunnel installation recipe is supplied without a verified supported workflow.

## Optional HTTP demo procedure

After an approved/provider-supported host exists: complete local native correctness and readiness checks, configure a session-specific bearer token, bind the application to loopback, and forward only its HTTP port with the selected tunnel. Record provider, transport and configuration in the serving manifest. Share endpoint/token only with intended testers through their normal channel; never commit credentials.

Use the same request schema, queue limits and cancellation semantics as localhost. Disable prompt/output logging except the public evaluation fixtures. Do not expose notebook files, arbitrary shell execution or model-download selection through the API. Stop the tunnel/server at the end of the test, revoke the session token and export results. Reconnects create a new deployment/run identity; clients cannot assume the old endpoint still owns an active request.

Tunnel measurements include network variation. Compare local kernel/model results independently and never infer a kernel change from one faster remote response.

## Operational behavior

On startup failure, retain logs and return not-ready; do not silently serve reference mode under a native label. On request OOM or a context-compromising CUDA fault, fail the request, stop admission and restart/revalidate before serving again. Graceful shutdown cancels queued work, signals active work, then stops the process after the documented grace.

Rollback means start a separately configured reference-mode server using the same model revision and API contract; identify that backend in metrics and result manifests. Do not mutate a running model's patch state while a request is active. Full-model weight rollback is not required because adaptation never modifies weights.

Future production operation requires a supported host, resource budgets, monitored latency/error objectives and a maintained dependency update process. None is implied by a successful short notebook demonstration.
