# Browser torch-webgpu architecture

## Decision

The wheel uses the Dawn-style C++ API from Emdawnwebgpu, but not Emdawn's full
JavaScript port. The exact Emdawn package and torch-webgpu source snapshot are
vendored and pinned in `config/build.toml`; ordinary wheel builds perform no
network fetch for either source.

PyTorch's `torch._C` is a WebAssembly side module in stock Pyodide. Emscripten
5.0.3 exports an `EM_JS` function body in the side module's `__em_js__*`
metadata. Pyodide's main-module dynamic loader reconstructs and evaluates that
body when it loads the wheel. The existing add/multiply backend demonstrated
that path in a real browser.

This depends specifically on Pyodide's `SIDE_MODULE=1` link profile. The
repository therefore keeps `pyodide build --exports=whole_archive`: in
pyodide-build 0.36.0 that leaves the configured `SIDE_MODULE=1` flag intact.
An exact Emscripten 5.0.3 probe compiled this bridge as a side module, loaded it
from a `MAIN_MODULE=1` process, created a buffer through
`wgpuDeviceCreateBuffer`, and observed the `EM_JS` allocation and final-release
destruction. Repeating the probe with pyodide-build's optimized
`SIDE_MODULE=2` profile left the internal `twgpu_*` imports unresolved. Changing
the wheel export mode is therefore an explicit compatibility boundary.

The resulting PyTorch wheel was also loaded by unmodified Pyodide 314.0.2 in
Chromium. The browser test initialized a real `GPUDevice`, exercised buffer
lifetime and copies, and numerically verified the imported add, multiply,
ReLU, 2-D matrix-multiplication, and last-dimension softmax kernels. Diagnostic
counters recorded nine dispatches and zero CPU fallbacks. This validates the
whole packaging path, rather than only the standalone loader probe.

Full Emdawnwebgpu is different. Its `webgpu.cpp` layer calls `emwgpu*` symbols
implemented by four `library_webgpu*.js` files. Emscripten processes those
files only while final-linking a main module and intentionally omits
JavaScript-library contents from a side module. Stock Pyodide therefore cannot
satisfy those imports. Requiring the complete port would require rebuilding
Pyodide, contrary to this project's compatibility boundary.

The implemented compromise is a compute-only Emdawn profile:

- compile against the exact Emdawn `webgpu.h` and `webgpu_cpp.h`;
- implement only the Dawn C entry points reached by the selected eager
  kernels;
- carry each browser implementation in side-module `EM_JS`;
- leave every unused C entry point undefined so newly imported upstream code
  fails at link time rather than silently acquiring unsupported behavior.

## Initialization and handles

`await torch.webgpu.init()` requests a browser `GPUAdapter`/`GPUDevice`, or
accepts a device supplied by a test harness. It creates one JavaScript state
object on `globalThis.__torchWebGPU` and reserves handle `1` for the device and
handle `2` for its queue. Only then does it call the native
`webgpu::initialize` operator, which adopts those handles as `wgpu::Device`
and `wgpu::Queue` objects.

Every other Dawn object receives a monotonically increasing 32-bit handle.
The browser object, type, reference count, and optional buffer size live in a
JavaScript `Map`. C++ `wgpu::*` wrappers retain and release those handles using
their normal RAII behavior. The last buffer release removes both registry
entries and calls `GPUBuffer.destroy()`. Imported device/queue objects remain
owned by JavaScript.

PyTorch storage contains a C++ `WebGPUAllocation` holding a `wgpu::Buffer`.
Destroying the storage releases exactly one owned buffer reference. Submitted
commands retain their WebGPU resources according to browser WebGPU lifetime
rules, so releasing the host handle after submission is safe.

## Commands, memory, and readback

The compatibility profile supports buffer creation/destruction, queue writes,
buffer copies, WGSL shader modules, bind-group/pipeline layouts, compute
pipelines, bind groups, command encoders, compute passes, dispatch, finish, and
submit. It does not expose adapter/device requests, mapping, futures, rendering,
textures, or native `WaitAny`.

CPU-to-GPU upload calls `GPUQueue.writeBuffer` synchronously with the current
Emscripten `HEAPU8`. Pyodide allows Wasm memory growth, so no heap view or Wasm
pointer is cached across an `await`. GPU-to-CPU remains the explicit Python
coroutine `torch.webgpu.to_cpu_async`: it copies into a MAP_READ buffer, awaits
`mapAsync`, copies the mapped bytes, and then unmaps and destroys the temporary
buffer. Synchronous `.cpu()` and `.item()` fail with a directed error.

## Imported operator policy

The build explicitly lists its upstream source files. It does not use
torch-webgpu's glob-based extension build because that would register native
`WaitAny` readback and several synchronous CPU fallbacks.

The compiled upstream helpers are:

- binary TensorIterator dispatch and its add/multiply/subtract/divide WGSL;
- unary TensorIterator dispatch and activation/trigonometric WGSL;
- scalar-power WGSL.

Matrix multiplication embeds the executable body of the pinned `mm.wgsl`
instead of performing upstream's runtime filesystem read. Softmax embeds the
pinned reference shader and is restricted to contiguous float32 tensors along
the last dimension. Unsupported dtypes, mixed CPU/WebGPU inputs, autograd,
and fallback-dependent operators fail before dispatch.

## Diagnostics and fragile dependencies

`torch.webgpu.diagnostics()` reports buffer allocations/releases, uploads,
GPU copies, WGSL shader compilations, pipeline creations, command submissions,
dispatches, readbacks, failed dispatches, and CPU fallbacks. Kernel caches are
held by the imported C++ implementations; the browser suite verifies that a
second invocation does not recompile the shader.

The architecture depends on behavior verified in Emscripten 5.0.3's dynamic
loader: evaluation of `SIDE_MODULE=1` `__em_js__` exports. It also requires a
content-security policy that permits that dynamic evaluation. Both the
Emscripten version and Emdawn release are pinned because `webgpu_cpp.h` is not
an ABI-stable interface. A future switch to `--exports=pyinit` or `requested`
must first solve or reproduce the `SIDE_MODULE=2` unresolved-import behavior.
