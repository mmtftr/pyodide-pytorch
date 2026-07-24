# Upstream CPU/Wasm test policy

The build runs upstream PyTorch tests from the exact source commit in
`config/build.toml`. `tests/upstream_cpu_wasm.json` is the executable source of
truth: `tests/upstream.mjs` verifies the checkout commit and rejects local
changes to the named test files, copies those files without modifying them,
installs the built wheel in the pinned Pyodide runtime, and executes the
generated test IDs through
`tests/run_upstream_tests.py`.

The current contract is:

| Upstream module | Passing tests |
| --- | ---: |
| `test/test_comparison_utils.py` | 6 |
| `test/test_type_info.py` | 5 |
| `test/test_numpy_interop.py` | 43 |
| `test/test_as_strided.py` | 2 |
| `test/test_module_tracker.py` | 3 |
| `test/test_complex.py` | 15 |
| `test/test_shape_ops.py` | 86 |
| `test/test_type_promotion.py` | 423 |
| `test/test_linalg.py` | 71 |
| **Total** | **654** |

CI requires all 654 tests to pass. It also requires zero runtime skips, expected
failures, failures, errors, and unexpected successes. A changed generated test
count, missing exact test ID, unknown exclusion, duplicate selection, source
commit mismatch, torch version mismatch, or wheel commit mismatch fails the
gate.

## Explicitly excluded generated tests

Whole-module selections use `["*"]`; every omitted generated test must appear
under `excluded_tests` with a nonempty reason. The current 16 exclusions are:

| Upstream test ID | Reason |
| --- | --- |
| `TestNumPyInteropCPU.test_copy_mode_cpu` | Requires `torch.compile` and `functorch`; this CPU/Wasm wheel is intentionally built with `BUILD_FUNCTORCH=0`. |
| `TestNumPyInteropCPU.test_ndarray_astype_object_graph_break_2_cpu` | Requires `torch.compile` and `functorch`; this CPU/Wasm wheel is intentionally built with `BUILD_FUNCTORCH=0`. |
| `TestNumPyInteropCPU.test_ndarray_astype_object_graph_break_cpu` | Requires `torch.compile` and `functorch`; this CPU/Wasm wheel is intentionally built with `BUILD_FUNCTORCH=0`. |
| `TestShapeOpsCPU.test_flip_large_tensor_cpu` | Upstream marks this accelerator-only because CPU execution is too slow; its decorators request 17 GB and 81 GB stress configurations. |
| `TestShapeOpsCPU.test_flip_unsupported_dtype_cpu_quint2x4` | Marked `expectedFailure` upstream for an unsupported packed quantized dtype, so it cannot contribute a passing assertion to this gate. |
| `TestShapeOpsCPU.test_flip_unsupported_dtype_cpu_quint4x2` | Marked `expectedFailure` upstream for an unsupported packed quantized dtype, so it cannot contribute a passing assertion to this gate. |
| `TestShapeOpsCPU.test_nonzero_cpu_bfloat16` | Pyodide NumPy produces 32-bit `intp` indices on wasm32, while `torch.nonzero` produces `torch.int64`. This unmodified upstream test requires equal dtypes and fails at that comparison. |
| `TestShapeOpsCPU.test_nonzero_cpu_bool` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_float16` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_float32` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_float64` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_int16` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_int32` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_int64` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_int8` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |
| `TestShapeOpsCPU.test_nonzero_cpu_uint8` | Same wasm32 NumPy `intp` versus PyTorch `int64` reference-dtype mismatch. |

The other upstream `nonzero` cases remain enabled:
`test_nonzero_astuple_out_cpu`, `test_nonzero_discontiguous_cpu`,
`test_nonzero_no_warning_cpu`, and `test_nonzero_non_diff_cpu`. The source is
not patched to weaken dtype comparison.

## Collection-only accommodation

`test/test_linalg.py` imports three quantization helpers at module scope.
PyTorch's `common_quantization.py` then imports
`functorch.experimental.control_flow`, even though none of the 71 selected
linalg tests uses those quantization helpers. Since `BUILD_FUNCTORCH=0`, the
runner installs an empty collection-only module for that exact name.

The accommodation cannot make a selected test silently pass: the empty module
has no API, so any attempted use fails. The runner also first verifies that the
target is genuinely unavailable and refuses to hide it if a future build makes
it importable.

## Probed but not admitted to the CI contract

These cases were evaluated while hill-climbing the suite but are not counted as
passing or as manifest exclusions:

| Test or module | Decision |
| --- | --- |
| `test/test_pytree.py` | Collection stops because `torch.utils._cxx_pytree` requires optional `optree>=0.13.0`, which is absent from the pinned Pyodide environment. The file imports the C++ implementation before Python-only cases can be selected. No fake `optree` implementation is installed. |
| `TestLinalgCPU.test_eigh_lwork_lapack_cpu_complex128` | Uses a 3000×3000 eigendecomposition. The first complex128 probe remained in the solver beyond a minute, so it is unsuitable for the per-build browser gate. |
| `TestLinalgCPU.test_eigh_lwork_lapack_cpu_complex64` | Same 3000×3000 stress case. Smaller upstream `eigh` cases for this dtype are enabled. |
| `TestLinalgCPU.test_eigh_lwork_lapack_cpu_float32` | Same 3000×3000 stress case. Smaller upstream `eigh` cases for this dtype are enabled. |
| `TestLinalgCPU.test_eigh_lwork_lapack_cpu_float64` | Same 3000×3000 stress case. Smaller upstream `eigh` cases for this dtype are enabled. |
| `TestLinalgCPU.test_linalg_lstsq_cpu_float32` | Passed locally, but the vendored single-precision `SGELSD` routine emits excessive diagnostic output. CI uses `test_linalg_lstsq_batch_broadcasting_cpu_float32` for quiet single-precision coverage and keeps the full least-squares test for float64, complex64, and complex128. |

The linalg module is an exact positive selection rather than a whole-module
selection because the file also contains accelerator, multi-GPU, tunable-op,
large-memory, compile, and backend-specific tests. Tests not named in the
manifest are outside the current asserted scope; they are not represented as
passing. Expanding the suite requires adding exact generated IDs, running them
against the pinned wheel, and updating the expected count.
