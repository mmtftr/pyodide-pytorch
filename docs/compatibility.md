# Compatibility and scope

Each wheel is tied to one CPython/Emscripten ABI. The initial baseline is:

| Component | Pin |
| --- | --- |
| PyTorch | `7bcf7da3a268b435777fe87c7794c382f444e86d` (`2.1.0a0`) |
| Pyodide | `0.24.1` |
| CPython | `3.11.2` / `cp311` |
| Emscripten | `3.1.45` |
| Wheel | `0.45.1` (last release providing `wheel.cli` for the pinned auditwheel plugin) |
| Ninja | `1.13.0` |

The wheel is CPU-only and intentionally has no WebAssembly shared memory. Both
ATen thread counts are fixed at one, and inter-op work runs inline. This avoids
the cross-origin isolation and worker requirements of pthread-enabled Wasm.

Unsupported or intentionally omitted areas include CUDA/ROCm, distributed
training, multiprocessing/shared-memory tensors, OpenMP, MKL/MKLDNN, NNPACK,
QNNPACK, XNNPACK, FBGEMM, Kineto, standalone functorch, C++ extension builds,
and command-line programs such as `torchrun`. APIs in those areas may still be
importable but are not functional.

Pyodide does not promise that a wheel built for one raw Emscripten version will
work with another. A new Pyodide ABI therefore requires a new build entry and a
fresh runtime test; it must not reuse or rename an old wheel.
