# Compatibility and scope

Each wheel is tied to one CPython/Emscripten ABI. The pinned upgrade target is:

| Component | Pin |
| --- | --- |
| PyTorch | `cf30153c4c131c8164ee7798e5022d810682e2cb` (`2.13.0`) |
| Pyodide | `314.0.2` |
| pyodide-build | `0.36.0` |
| CPython | `3.14.2` / `cp314` |
| Platform tag | `pyemscripten_2026_0_wasm32` |
| Emscripten | `5.0.3` |
| Wheel | `0.47.0` |
| Ninja | `1.13.0` |
| CMake | `3.27.9` (satisfies PyTorch 2.13's CMake 3.27 minimum) |

Only patch applicability against this exact PyTorch commit has been validated
at this revision. The target remains provisional until the canonical CI full
build passes binary validation and the runtime smoke suite in Pyodide 314.0.2.
The build explicitly mirrors PyTorch 2.13's C++20 and C17 language-standard
requirements. Exception handling follows the Pyodide 314 ABI
(`-fwasm-exceptions` with Wasm `longjmp` support); the patch series removes
PyTorch's incompatible legacy `DISABLE_EXCEPTION_CATCHING` setting.

The wheel is CPU-only and intentionally has no WebAssembly shared memory. Both
ATen thread counts are fixed at one, and inter-op work runs inline. Requests to
set either count above one raise `RuntimeError`. This avoids the cross-origin
isolation and worker requirements of pthread-enabled Wasm.

Unsupported or intentionally omitted areas include CUDA/ROCm, distributed
training, multiprocessing/shared-memory tensors, OpenMP, MKL/MKLDNN, NNPACK,
QNNPACK, XNNPACK, FBGEMM, Kineto, standalone functorch, C++ extension builds,
and command-line programs such as `torchrun`. APIs in those areas may still be
importable but are not functional.

The `pyemscripten` platform ABI is the compatibility boundary; the raw
Emscripten version alone is not a wheel tag. A new Pyodide ABI therefore
requires a new build entry and a fresh runtime test; it must not reuse or
rename an old wheel.
