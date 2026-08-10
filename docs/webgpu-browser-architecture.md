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

The resulting PyTorch wheel is loaded by unmodified Pyodide 314.0.2 in
Chromium. The browser test initializes a real `GPUDevice`, exercises buffer
lifetime and copies, and numerically verifies isolated eager kernels plus a
complete tiny decoder block. Diagnostic counters and a mandatory zero
CPU-fallback count validate the whole packaging path rather than only the
standalone loader probe.

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

`torch.webgpu.init_async()` requests a browser `GPUAdapter`/`GPUDevice`, or
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
pointer is cached across an `await`. GPU-to-CPU has the explicit Python
coroutine `torch.webgpu.to_cpu_async`: it copies into a MAP_READ buffer, awaits
`mapAsync`, copies the mapped bytes, and then unmaps and destroys the temporary
buffer. The CPU tensor is built with `torch.frombuffer(...).clone()` rather
than `torch.from_numpy()`. This avoids the Pyodide/PyTorch bridge regression
that raised `element_size must be 0`, and the clone gives PyTorch-owned storage
after the temporary NumPy array is collected.

Synchronous `torch.webgpu.init()`, `torch.webgpu.synchronize()`,
`torch.webgpu.to_cpu_sync`, WebGPU `Tensor.cpu()`, and WebGPU `Tensor.item()`
use Pyodide's JSPI `run_sync` bridge around the corresponding coroutine. They
work only
when `pyodide.ffi.can_run_sync()` is true and Python was entered through
`runPythonAsync()` or a PyProxy `callPromising()` call. `runPython()` and a
direct synchronous PyProxy call receive a directed error naming both valid
entrypoints and the async fallback. Ordinary CPU tensors still delegate to
PyTorch's original `cpu()` and `item()` methods.

JSPI must be feature-detected rather than inferred from a user agent. Chrome
137 and later ship JSPI without a browser flag; Node 24 still needs
`--experimental-wasm-jspi`. Runtimes without stack switching must use
`await torch.webgpu.init_async()`, `await torch.webgpu.synchronize_async()`, or
`await torch.webgpu.to_cpu_async(tensor)`. See Pyodide's
[JSPI overview](https://blog.pyodide.org/posts/jspi/) and
[`run_sync` API](https://pyodide.org/en/stable/usage/api/python-api/ffi.html#pyodide.ffi.run_sync).

Float32 activations and int32 token IDs use the same raw 32-bit buffer path.
Restricted Long token/position tensors retain normal 8-byte ATen storage and
use two adjacent 32-bit words per logical element; the high word must be the
sign extension of a signed-int32 low word.
The tensor dtype remains native PyTorch metadata; every operator checks the
expected dtype before interpreting those bytes. Bool retains ATen's one-byte
storage contract. WGSL sees the same bytes as packed `u32` words, and Bool
kernels assign one invocation per destination word (or serialize arbitrary
strided writes) so two invocations never race on adjacent bytes.

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

Decoder kernels that are not present as a complete browser-safe upstream path
live in the project-owned `webgpu/llm_kernels` directory, never under
`vendor/`. The staging script copies that directory beside the two pinned
source trees and generates a C++ header containing its WGSL. The wheel thus
performs no runtime shader-file reads and no network fetch. The initial set is
strided copy/concatenation, int32 embedding, batched matrix multiplication,
LayerNorm, RMSNorm, last-dimension greedy argmax, and fused causal or
float-mask SDPA with GQA head mapping.
The LayerNorm path flattens an arbitrary nonempty normalized suffix after
GPU-only contiguous materialization (ranks up to eight), then assigns one
workgroup to each prefix row. Its WGSL combines Welford `(count, mean, M2)`
states instead of subtracting two large moments. It writes the native output,
mean, and reciprocal-standard-deviation tuple in one dispatch and binds absent
affine inputs only through read-only aliases, avoiding WebGPU's same-buffer
read/write validation conflict. Empty prefix rows validate metadata and return
without encoding a command.
Rotary encoding in the browser test is composed from slice, negate,
concatenation, multiply, and add instead of introducing a model-specific
kernel.

Pipeline objects are function-local C++ statics. Dawn-style RAII retains their
numeric handles for the process lifetime and releases temporary bind groups,
parameter buffers, encoders, and command buffers after submission. WebGPU's
submission lifetime rules keep referenced browser resources alive until the
queued work completes.

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

## Decoder verification snapshot

The pinned development wheel was built with CPython 3.14.2, Emscripten 5.0.3,
and the stock Pyodide 314.0.2 xbuild environment. Chromium's SwiftShader
WebGPU adapter completed every numerical comparison in `tests/webgpu.html`,
including GQA, additive-mask attention, composed rotary encoding, and the
batch-one, eight-token tiny GPT block. The run recorded 167 compute dispatches,
one explicit GPU copy, 25 asynchronous readbacks, 14 shader compilations and
pipeline creations, zero failed dispatches, zero CPU fallbacks, and zero
WebGPU validation errors.

The five-repeat timing sample was 11.22 ms per decoder forward, or about 713
tokens/second, versus 0.96 ms for CPU PyTorch in the same browser. These tiny
shapes are dominated by dispatch overhead and SwiftShader is a software
adapter, so the values are a regression diagnostic rather than a hardware GPU
performance claim. Wheel validation, the CPU/Pyodide smoke test, and all 654
selected upstream CPU tests also passed.
