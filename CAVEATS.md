# Caveats and unsupported features

This project makes a useful subset of PyTorch available in Pyodide. It does not
turn the browser into a fully compatible native PyTorch environment. Treat the
tested surface as an explicit lower bound, not as a claim that every importable
API works.

## Project status

- The wheels are experimental, unofficial downstream builds.
- PyTorch and Pyodide do not publish or support these release artifacts.
- Bug reports should include the exact release tag, browser or Node version,
  Pyodide version, and a minimal reproduction.
- PyTorch, Pyodide, browser, or WebAssembly defects should be reported to the
  corresponding upstream project when they are not specific to this build.

See [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## CPU and threading

The CPU execution path is deliberately threadless:

- intra-op and inter-op thread counts are fixed at one;
- attempts to set either count above one raise `RuntimeError`;
- Emscripten pthreads, WebAssembly shared memory, and the atomics target feature
  are absent;
- ATen scheduling and CPU autograd callbacks execute synchronously on the
  thread running Python.

The playground runs Pyodide in a Web Worker so computation does not block its
interface. An application that runs Pyodide on the browser's main thread can
still freeze its own UI during PyTorch operations.

The threadless build does not require the cross-origin isolation headers needed
for shared WebAssembly memory.

## Experimental WebGPU backend

The `r4` release and development wheels built from `main` include an
experimental `PrivateUse1` backend named `webgpu`. It submits real WGSL
compute work through the browser WebGPU API, but it is not a general PyTorch
accelerator backend.

Release `r4` is limited to one device, `torch.float32`, CPU-to-GPU and
GPU-to-GPU copies, addition, multiplication, broadcasting, metadata-only
views, and explicit asynchronous readback. The current development wheel also
browser-verifies ReLU, 2-D matrix multiplication, and contiguous last-dimension
softmax through the pinned torch-webgpu WGSL implementations. Other imported
entry points remain compile-only until individually exercised. Consult the
[operator support table](docs/webgpu-operator-support.md). Unsupported
operators raise errors; there is no implicit CPU fallback.

Browser GPU-to-CPU transfer requires `GPUBuffer.mapAsync()`. Stock
single-threaded Pyodide cannot turn that Promise into a synchronous PyTorch
copy, so `.cpu()`, `.item()`, and operations that need to inspect values on the
host do not work for WebGPU tensors. Use
`await torch.webgpu.to_cpu_async(tensor)`.

The WebGPU backend:

- requires a browser with WebGPU enabled and a secure context outside
  localhost;
- is separate from CUDA, so `torch.cuda.is_available()` remains false;
- does not support autograd, modules, optimizers, general reductions, or model
  inference beyond explicitly registered and browser-verified operators;
- uses 32-bit shape, stride, and storage-offset metadata and supports at most
  eight dimensions;
- rejects copies between different views of the same `GPUBuffer`; exact-alias
  copies are no-ops and copies between distinct buffers are supported;
- has only been validated with the repository's deterministic SwiftShader
  browser test so far.

The side-module/Dawn compatibility design, JavaScript packaging, object
lifetimes, memory-growth rules, and fragile dependencies are documented in
[`docs/webgpu-browser-architecture.md`](docs/webgpu-browser-architecture.md).

## Performance and memory

Expect lower performance than native PyTorch:

- execution is single-threaded;
- the initial runtime download is tens of MiB;
- WebAssembly compilation and Python package initialization add startup cost;
- browser and WebAssembly memory limits make large models and large
  decompositions impractical even when an operation is implemented.

Browsers may evict cached assets according to their own storage policies. The
playground's **Clear cache** control removes only caches created by this
project.

## ABI compatibility

Each wheel is tied to an exact Pyodide platform ABI, CPython tag, and
WebAssembly target:

- do not load it in a different Pyodide version unless that version has the
  same explicitly tested ABI;
- do not rename a wheel to claim compatibility with a different runtime;
- do not install it into native Linux, macOS, or Windows CPython;
- use the wheel filename and runtime version recorded in
  `build-manifest.json`.

Pyodide's `pyemscripten` platform ABI is the compatibility boundary. Matching
only the raw Emscripten version is insufficient. See
[`docs/compatibility.md`](docs/compatibility.md) for the current pins.

## Unsupported subsystems

The current build intentionally omits:

- CUDA, ROCm, MPS, XPU, and accelerator backends other than the narrow
  experimental WebGPU implementation described above;
- distributed training and RPC;
- multiprocessing and filesystem-backed shared-memory tensors;
- OpenMP, MKL, MKLDNN, FBGEMM, XNNPACK, QNNPACK, and NNPACK;
- Kineto and native profiler components;
- `torch.compile`, TorchDynamo, and the standalone functorch build;
- runtime C++ or CUDA extension compilation;
- command-line programs such as `torchrun`.

An API may remain importable because it is part of PyTorch's Python package
surface even when its native backend was not built. Importability is not a
support guarantee.

Selected `torch.func` operations pass the repository smoke suite despite the
standalone functorch and `torch.compile` stack being disabled. Consult the
tests before relying on an unlisted transform.

## Linear algebra

The wheel vendors the `libopenblas.so` side module from the matching Pyodide
release and uses it for LAPACK-backed `torch.linalg` operations. The build
adapts PyTorch's Fortran declarations to Pyodide's `f2c` calling convention:
the relevant subroutines have an `i32` WebAssembly result instead of the
`void` result used by native Fortran ABIs.

Runtime smoke tests cover inverse, solve, eigenvalue, and Cholesky operations.
The upstream gate includes 71 real and complex, single- and double-precision
linalg tests. This is substantial coverage, but it is not every
`torch.linalg` operation, dtype, input size, or error path.

## NumPy index width on wasm32

NumPy's `intp` is 32-bit on wasm32, while PyTorch correctly returns
`torch.int64` indices from `torch.nonzero`. Ten unmodified upstream tests
compare those cross-library dtypes and are explicitly excluded for that
reference mismatch. Other upstream `nonzero` tests remain enabled, and the
PyTorch source is not patched to weaken the dtype comparison.

The exact exclusions and rationale are recorded in
[`docs/upstream-tests.md`](docs/upstream-tests.md).

## Test coverage

CI runs 654 selected tests copied verbatim from the pinned PyTorch source
commit. It requires all selected tests to pass with no runtime skips or
expected failures.

This does not imply that the complete PyTorch test suite passes in Pyodide.
Accelerator-specific, distributed, compile, large-memory, and other unsupported
areas are outside the asserted scope. The executable selection lives in
[`tests/upstream_cpu_wasm.json`](tests/upstream_cpu_wasm.json).

## Distribution and trust

Loading a wheel executes Python and WebAssembly code in the Pyodide runtime.
For deployments:

1. pin an immutable GitHub release rather than following the mutable playground
   manifest;
2. verify the wheel against its `.sha256` file and `build-manifest.json`;
3. serve the verified artifacts from an origin with an appropriate CORS
   policy;
4. retain the release manifest so the PyTorch source commit and complete ABI
   tuple remain auditable.

Release artifacts are generated by GitHub Actions and accompanied by build
provenance when the repository configuration supports attestations.
