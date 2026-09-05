# Continuous integration and documentation delivery

Status: documentation/security workflows and F01 package CI implemented; hosted-run outcomes are recorded separately. Required T4 qualification is pending; native RMSNorm and inference tests remain planned.

## Checks active at this stage

| Check | Trigger | Meaning |
|---|---|---|
| Documentation validator | Branch push, PR, manual | Local link targets, balanced fences, JSON/SSE example syntax, requirement/test references, workflow conventions and credential-shaped content |
| Validator unit tests | Same | Regression cases prove malformed examples, missing links and unsafe workflow conventions are detected |
| Ruff lint/format | Same | Python repository-tooling quality |
| pip check | Same | Installed CI dependency compatibility |
| pip-audit | Push, PR, weekly, manual | Known vulnerabilities in declared tooling dependencies and resolved transitives; findings fail the job |
| Dependency review | PR | Newly introduced known vulnerabilities at moderate severity or above block the check |
| CodeQL Python | Push, PR, weekly, manual | Security analysis of actual Python validation tools/tests; not CUDA analysis |
| CodeQL C/C++ | Same | Build-free analysis of host binding source; incomplete dependency/type inference is possible and `.cu` execution is not certified |
| Dependabot | Weekly configuration; update PR limit currently zero | Maintainer-controlled updates preserve the one-open-PR rule; dependency alerts and audits remain active |
| Documentation bundle | After documentation/tooling matrix passes | Downloadable ZIP artifact of README and doc, retained 14 days |
| Package/reference | Linux 3.11 and Windows 3.12 | Build sdist/wheel, inspect installed wheel, test supporting code and notebook syntax; GPU tests explicitly deselected |
| Package dependency audits | Package jobs | Audit installed transitives and canonical direct pins; the latter covers the public PyTorch release when its `+cpu` wheel label is not found by the advisory service |
| Package artifacts | Package jobs, including failure | Distribution files, JUnit and audit reports; no compiled CUDA wheel or GPU qualification claim |

CI runs Linux/Python 3.11 and Windows/Python 3.12. All third-party actions are pinned to full commit SHAs with version comments; updates require review. Dependabot version-update PRs are paused with a zero limit so automation cannot open a second PR while feature work is active. Enable/update one dependency change only when the PR queue is empty, then restore the limit. Dependency alerts and vulnerability audits continue. Checkout credentials are not persisted. Default workflow permissions are read-only; CodeQL alone receives security-events write. No personal token is stored in workflows or required by normal CI. PR jobs use `pull_request`, not privileged `pull_request_target` execution.

The repository is public, so the selected security checks target public-repository capabilities. Dependency graph/security settings can still affect availability; a failed/unavailable hosted check must be investigated, not converted to a passing result. Scheduled runs and Dependabot require configuration on the default branch. Open only one PR at a time, wait for green checks, merge, then branch again from updated main.

## Local validation

Create a dedicated Python environment, install `requirements-ci.txt`, then run:

```text
python -m pip check
python -m ruff check tools tests
python -m ruff format --check tools tests
python -m unittest discover -s tests -p "test_*.py" -v
python tools/check_docs.py
python -m pip_audit -r requirements-ci.txt
git diff --check
```

The validator checks local targets, not remote link availability or Markdown fragment anchors. Its credential check covers common token/key shapes; it is not a comprehensive secret-scanning service. Mermaid diagrams are kept as readable Markdown source; diagram rendering and semantic GPU behavior are not certified by these syntax checks.

## Delivery and future gates

The current CD output is a validated documentation artifact, not an automatic deployment to a GPU session, package registry or public website. The bundle waits for documentation/tooling checks; security workflows report independently and must also pass before merge/release. No inference binary is published before code exists.

F01 implements supplementary package build/install and CPU reference checks in hosted CI, plus a required contributor-operated [T4 notebook](../../notebooks/01-t4-build-smoke.ipynb) for native compilation/execution and CUDA-tensor reference checks. Its actual T4 run remains pending. Native builds happen inside the Linux GPU notebook environment, not on the contributor's local machine. CPU package checks do not certify the CUDA extension. Record the tested commit and environment with the GPU logs; green CPU CI alone does not complete F01.

F02 extends the GPU notebook with RMSNorm numerical/stream checks and sanitizer evidence. GPU testing must not execute untrusted PR code with credentials on a persistent self-hosted runner. F05/F08 add real model and API functionality; F13 adds release artifact verification and explicit publishing gates. F01 adds [build-free C/C++ CodeQL](https://docs.github.com/en/code-security/reference/code-scanning/codeql/build-options-for-compiled-languages) for the actual binding source. This infers compilation context and may have incomplete type/header coverage; it does not build or certify the CUDA source and does not replace CUDA race/memory tests.

Recommended required checks once workflows are established on main: both documentation/tooling matrix jobs, dependency audit, dependency review on PRs, and CodeQL. Branch protection is not changed automatically by this task.

Sources checked 2026-09-05: [GitHub secure workflow use](https://docs.github.com/en/actions/reference/security/secure-use), [dependency review](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/manage-your-dependency-security/configure-dependency-review-action), [CodeQL action](https://github.com/github/codeql-action), [pip-audit](https://github.com/pypa/pip-audit), [Dependabot configuration](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference).
