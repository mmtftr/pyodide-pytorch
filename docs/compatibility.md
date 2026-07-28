# Compatibility and tested scope

Every wheel is built for one exact CPython and Pyodide WebAssembly ABI. The
current release uses the following tuple:

| Component | Pin |
| --- | --- |
| PyTorch source | `cf30153c4c131c8164ee7798e5022d810682e2cb` (`2.13.0`) |
| Wheel version | `2.13.0+pyodide314.0.2.r4` |
| Pyodide | `314.0.2` |
| `pyodide-build` | `0.36.0` |
| CPython | `3.14.2` / `cp314` |
| Platform tag | `pyemscripten_2026_0_wasm32` |
| Emscripten | `5.0.3` |
| LAPACK | Pyodide `libopenblas` `0.3.28` |
| `auditwheel-emscripten` | `0.2.5` |
| Wheel | `0.47.0` |
| Ninja | `1.13.0` |
| CMake | `3.27.9` |
| Release | `torch-2.13.0-pyodide-314.0.2-r4` |

[`config/build.toml`](../config/build.toml) is the machine-readable source of
truth. This document describes the release for humans and must be updated when
that configuration changes.

## Compatibility boundary

The `pyemscripten` platform tag is the compatibility boundary. Matching the
raw Emscripten version alone does not establish compatibility because Pyodide
versions can differ in CPython, linked side modules, compiler flags, and
platform ABI.

A wheel must be rebuilt and retested when any ABI-relevant member of the tuple
changes. It must not be renamed or loaded into a different native or
WebAssembly Python runtime on the assumption that the import tags are close
enough.

The build uses PyTorch 2.13's C++20 and C17 language standards. Exception
handling follows the Pyodide 314 ABI (`-fwasm-exceptions` with WebAssembly
`longjmp` support); the patch series removes PyTorch's incompatible legacy
`DISABLE_EXCEPTION_CATCHING` setting.

## Tested runtime scope

Release `torch-2.13.0-pyodide-314.0.2-r4` passed:

- wheel metadata and WebAssembly binary validation;
- the repository runtime smoke suite;
- 654 selected upstream PyTorch CPU tests;
- 71 selected LAPACK-backed `torch.linalg` tests across real and complex,
  single- and double-precision dtypes;
- a real Chromium WebGPU suite covering add, multiply, broadcast, strided
  views, copies, and asynchronous readback with no validation errors or
  implicit CPU fallbacks;
- version, Emscripten platform, and single-thread invariants.

The selected upstream gate permits no runtime skips, expected failures,
failures, errors, or unexpected successes. See
[`docs/upstream-tests.md`](upstream-tests.md) for the exact inventory and all
explicit exclusions.

## Deliberate build constraints

The WebAssembly CPU path has no shared memory. Both ATen thread counts are
fixed at one, and inter-op work runs inline. The wheel also includes the
narrow experimental browser WebGPU backend described in the project caveats.

LAPACK uses Pyodide's `f2c` ABI: Fortran subroutines have an `i32` result rather
than the native Fortran `void` result. Although callers ignore that result,
WebAssembly includes it in the function type and rejects mismatched
declarations at link time.

Unsupported areas include accelerator backends other than the experimental
WebGPU subset, distributed training, multiprocessing, shared-memory tensors,
`torch.compile`, runtime C++ extensions, and multithreaded CPU execution. The
full user-visible limitation list is maintained in
[CAVEATS.md](../CAVEATS.md).
