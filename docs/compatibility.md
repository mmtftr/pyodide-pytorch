# Compatibility and scope

Each wheel is tied to one CPython/Emscripten ABI. The pinned upgrade target is:

| Component | Pin |
| --- | --- |
| PyTorch | `cf30153c4c131c8164ee7798e5022d810682e2cb` (`2.13.0`) |
| Pyodide | `0.24.1` |
| CPython | `3.11.2` / `cp311` |
| Emscripten | `3.1.45` |
| Wheel | `0.45.1` (last release providing `wheel.cli` for the pinned auditwheel plugin) |
| Ninja | `1.13.0` |
| CMake | `3.27.9` (satisfies PyTorch 2.13's CMake 3.27 minimum) |

Only patch applicability against this exact PyTorch commit has been validated
at this revision. The target remains provisional until the canonical CI full
build passes binary validation and the runtime smoke suite in Pyodide 0.24.1.
The build explicitly mirrors PyTorch 2.13's C++20 and C17 language-standard
requirements.

The wheel is CPU-only and intentionally has no WebAssembly shared memory. Both
ATen thread counts are fixed at one, and inter-op work runs inline. Requests to
set either count above one raise `RuntimeError`. This avoids the cross-origin
isolation and worker requirements of pthread-enabled Wasm.

Unsupported or intentionally omitted areas include CUDA/ROCm, distributed
training, multiprocessing/shared-memory tensors, OpenMP, MKL/MKLDNN, NNPACK,
QNNPACK, XNNPACK, FBGEMM, Kineto, standalone functorch, C++ extension builds,
and command-line programs such as `torchrun`. APIs in those areas may still be
importable but are not functional.

Pyodide does not promise that a wheel built for one raw Emscripten version will
work with another. A new Pyodide ABI therefore requires a new build entry and a
fresh runtime test; it must not reuse or rename an old wheel.
