# F03: framework registration, step by step

Status: initial contributor T4 compilation and execution passed; see the [F03 review](../evaluation/07-f03-gpu-evidence-review.md). Implements REQ-K06 / T-K09; training and full-model compilation remain outside scope.

## 1. The mathematical operation stays the same

For each width-H row, compute `s = sum(float(x_i)^2) / H`, then `r = rsqrt(s + eps)`. Round `float(x_i) * r` to the input dtype before multiplying by the matching weight. The CUDA reduction and its FP16 rounding boundary are unchanged. Rerun all F02 numerical tests because changing dispatch can still break execution.

## 2. A schema describes the operation

`aegis_norm::rms_norm(Tensor x, Tensor weight, float eps) -> Tensor` promises one fresh output, no input mutation and no aliasing. Loading the native extension creates that schema and its CUDA implementation. Python then registers a metadata implementation before the loader reports success. Imports still do not compile anything. Registration is process-local; restart the process after changing loaded sources or after a partial registration failure.

## 3. FakeTensor describes the result without calculating it

Tracing systems need to know what shape and dtype an operator produces. `aegis_norm/registration.py` checks metadata and returns a fake contiguous tensor with the input shape, dtype and CUDA device. It reads no values and queries no GPU capability. Real execution retains the T4 capability check in C++.

`torch._check` expresses shape constraints, including symbolic widths and row counts. Width must be 1..65536; row count cannot exceed 2^31-1. Input and weight must be contiguous FP16 or FP32 on the same CUDA device. Epsilon must remain positive and finite after FP32 conversion. A nonzero input storage offset is allowed, but the output is newly allocated with offset zero.

## 4. Autograd must preserve dispatch

The Autograd handler rejects an active gradient requirement. Otherwise it excludes Autograd dispatch and calls the registered operator again, allowing PyTorch to choose CUDA execution or FakeTensor handling. Calling the CUDA launcher directly from this handler would bypass metadata tracing. No backward implementation is introduced.

## 5. Registration and numerical tests answer different questions

| Check | What it establishes |
|---|---|
| F02 numerical suite | Results match the cast-ordered reference within predefined tolerances |
| Metadata tests | Shape, dtype, device, contiguous output, nonaliasing and invalid-input behavior without GPU allocation |
| opcheck schema | Real execution follows its mutation and aliasing declaration |
| opcheck FakeTensor | Fake output metadata agrees with real execution |
| opcheck Autograd registration | Registration follows framework expectations for the supplied inference inputs |
| opcheck dynamic AOT dispatch | Operator behavior remains consistent through dynamic graph transformations |
| Explicit aot_eager smoke | Raw operator executes in a full graph across changing shapes and offset inputs |

The local metadata fixture uses an isolated test schema with a Python Autograd adapter. It does not load or qualify the real C++ adapter. T4 tests exercise the actual native registration. `aot_eager` avoids requesting a Triton-generated kernel; it is not a performance baseline. The public dispatch wrapper and a complete model are not covered by a compilation guarantee.

Supporting validation: 92 CPU tests pass, including 32 metadata cases and loader registration ordering/failure checks. Native GPU tests are deselected locally; the 10 T4 framework cases and 32 metadata cases subsequently passed in the contributor run.

## 6. Run and review evidence

Use [the F03 notebook](../../notebooks/03-t4-framework-registration.ipynb) with the feature commit's full SHA. Run every cell on T4. It exports fresh build evidence, the F02 regression JUnit report, `registration-tests.xml`, `registration-tests.log`, audit output and hashes in `aegis-f03-evidence.zip`. Review both success counts and skipped/failed cases before qualification. No timing in this suite is a benchmark.

Sources: [PyTorch custom C++ operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), [dispatcher and Autograd](https://docs.pytorch.org/tutorials/advanced/dispatcher), and [register_fake/opcheck contracts](https://docs.pytorch.org/docs/main/library.html). Checked against the selected PyTorch 2.14 environment; initial GPU registration results passed in the linked evidence.
