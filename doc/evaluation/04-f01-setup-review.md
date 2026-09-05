# F01 contributor setup log review

Reviewed 2026-09-05. Status: partial setup evidence; native compilation and GPU correctness remain pending.

## Observations

The contributor's console output identifies a Tesla T4, driver 580.82.07, a driver-reported CUDA maximum of 13.0 and an initially selected nvcc 12.8.93. The driver display is not the installed compiler version.

The [supplied installation log](logs/2026-09-05-toolkit-setup.txt) records installation of `cuda-toolkit-12-6`. Its final compiler checks report `nvcc` 12.6.85 and Ubuntu GCC 11.4.0. The apt source-index warning and ldconfig symlink warnings appear in the log, but the final compiler commands succeed. This establishes tool availability only; no extension build result follows from it. Post-install driver identity still needs the notebook preflight artifact.

The next supplied traceback reports Python 3.13 failing inside `ensurepip` during `venv.EnvBuilder(with_pip=True)`. Its underlying stderr is absent, so the exact cause is not proven. The notebook now installs the running interpreter's `pythonX.Y-venv` Ubuntu package, then creates/reuses the environment using a logged subprocess command. This addresses the usual missing version-specific pip-bootstrap support and retains diagnostics if creation still fails. See [Python venv behavior](https://docs.python.org/3/library/venv.html) and [Ubuntu's Python 3.13 venv package](https://packages.ubuntu.com/en/questing-updates/python/python3.13-venv); package availability in this session depends on its configured repositories.

## Changes and remaining evidence

- Explicitly select `/usr/local/cuda-12.6` through `CUDA_HOME`, `CUDACXX` and PATH, including an existing subprocess environment.
- Install the versioned toolkit alongside the provider toolkit when absent; no driver package is requested.
- Install the matching Python venv support and log environment creation before dependency installation.
- Preserve the partially created environment directory; no workspace deletion is required.

The repaired environment-creation cell has not yet been rerun by the contributor. Required follow-up evidence is pip availability, dependency installation/audits, preflight, native compilation, CUDA tests and the exported run archive. There is no benchmark or native correctness result yet.

The attached log was copied with trailing whitespace normalized for repository validation. Original attachment SHA-256: `ce9c67a1100fb057cf59382161daad6140ea2b395dad4bc374e0b0069b32fd3a`. It is a contributor console excerpt, not a complete run manifest or independently reproduced result.
