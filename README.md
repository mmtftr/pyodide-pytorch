# PyTorch for Pyodide

[![Validate repository](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/validate.yml)
[![Build PyTorch wheel](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/build.yml)
[![Deploy playground](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/pages.yml)
[![Latest release](https://img.shields.io/github/v/release/mmtftr/pyodide-pytorch?display_name=tag&sort=date)](https://github.com/mmtftr/pyodide-pytorch/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run PyTorch in a browser or another Pyodide environment. This repository
produces a reproducible WebAssembly wheel, tests it inside the exact Pyodide
runtime it targets, and publishes the wheel with checksums and build
provenance. The `r7` build contains an early browser WebGPU backend and a
release pipeline that promotes its exact browser-tested artifact to the
playground.

**[Try the browser playground](https://mmtftr.github.io/pyodide-pytorch/)**
· [Download the latest release](https://github.com/mmtftr/pyodide-pytorch/releases/latest)
· [Read the caveats](CAVEATS.md)

> [!IMPORTANT]
> This is an experimental downstream distribution. It is not an official
> PyTorch or Pyodide release.

## Current release

| Component | Version |
| --- | --- |
| PyTorch | `2.13.0+pyodide314.0.2.r7` |
| Pyodide | `314.0.2` |
| Python | `3.14.2` (`cp314`) |
| WebAssembly platform | `pyemscripten_2026_0_wasm32` |
| Release | `torch-2.13.0-pyodide-314.0.2-r7` |

The complete, ABI-relevant configuration lives in
[`config/build.toml`](config/build.toml). A wheel is compatible only with the
Pyodide ABI recorded in its release manifest; it is not a general-purpose
CPython wheel.

## Try it

The [playground](https://mmtftr.github.io/pyodide-pytorch/) loads the exact
release artifact promoted from a successful `main` build in a Web Worker. It
fails closed if the deployed manifest or wheel digest cannot be verified; it
never silently substitutes an older release. It provides:

- a CodeMirror Python editor and separate output console;
- autocompletion with `Tab` or `Ctrl+Space`;
- runtime-derived signatures and documentation on hover;
- examples for tensors, autograd, neural networks, optimization,
  `torch.linalg`, and CPU-versus-WebGPU elementwise and transformer decoder
  benchmarks;
- restart, cancellation, and versioned browser caching.

The first load downloads Pyodide, its Python dependencies, and the PyTorch
wheel. Compatible subsequent visits reuse the versioned runtime cache.

## Use it in a web application

The wheel is not published on PyPI. Load the matching Pyodide runtime and the
validated wheel deployed with the playground:

```html
<script type="module">
  const runtimeBase =
    "https://mmtftr.github.io/pyodide-pytorch/runtime/";
  const manifestResponse = await fetch(`${runtimeBase}build-manifest.json`);
  if (!manifestResponse.ok) {
    throw new Error(`Could not load release manifest: ${manifestResponse.status}`);
  }
  const manifest = await manifestResponse.json();

  const pyodideVersion = manifest.configuration.pyodide.version;
  const indexURL =
    `https://cdn.jsdelivr.net/pyodide/v${pyodideVersion}/full/`;
  const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL });

  await pyodide.loadPackage([
    "micropip",
    "numpy",
    "typing-extensions",
    "sympy",
    "networkx",
    "jinja2",
    "fsspec",
  ]);
  await pyodide.runPythonAsync(`
import micropip
await micropip.install("filelock==3.32.0")
`);

  const wheelURL = new URL(manifest.wheel.filename, runtimeBase);
  await pyodide.loadPackage(wheelURL.href);

  await pyodide.runPythonAsync(`
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x.square().sum().backward()
print(torch.__version__)
print(x.grad)
`);
</script>
```

This example follows the mutable playground manifest for convenience. For a
production deployment, pin an immutable
[release](https://github.com/mmtftr/pyodide-pytorch/releases), host the wheel
on a CORS-enabled origin, and verify its published SHA-256 digest.

## Experimental WebGPU backend

The `r7` release and development wheels built from `main` expose a real
PyTorch `webgpu` device in browsers that implement WebGPU:

```python
import torch

torch.webgpu.init()

x = torch.tensor([[1.0], [2.0]]).to("webgpu")
y = torch.tensor([[10.0, 20.0, 30.0]]).to("webgpu")
result = torch.add(x, y, alpha=2)

print(result.device)  # webgpu:0
print(result.cpu())
```

The backend is intentionally bounded:

- one browser `GPUDevice`, float32 model tensors, int32 token indices or
  signed-int32-valued indices in real 8-byte `torch.int64` storage, byte-packed
  Bool for the bounded mask-control path, and at most eight dimensions;
- GPU-native tensor addition and multiplication, including broadcasting,
  storage offsets, and `alpha` for addition;
- metadata-only `view`, slice, and transpose operations;
- CPU-to-GPU and GPU-to-GPU copies;
- JSPI-backed synchronous initialization, synchronization, `.cpu()`, `.item()`,
  and scalar truth, with `init_async`, `synchronize_async`, and
  `to_cpu_async` retained as explicit non-JSPI fallbacks;
- unsupported operations fail instead of silently falling back to the CPU.

The browser-verified decoder inference slice includes:
embedding, strided materialization/concatenation, BMM/linear, LayerNorm,
RMSNorm, fused causal/GQA attention, and a complete tiny GPT block assembled
with the existing activation and elementwise WGSL kernels. LayerNorm returns
the native output/mean/rstd tuple and uses a stable Welford reduction for
affine, multi-dimensional normalized suffixes. Linear projection and bias
addition share one tiled dispatch. An opt-in preallocated Qwen2/Llama/Mistral
cache writes paired K/V states in one indexed dispatch, and an explicit
decode-only Q8 linear path stores group-128 weights in 1.03125 bytes/value.
The separately
maintained kernels are under `webgpu/llm_kernels`, not the pinned vendor trees.
This does not yet include general cache implementations, sampling, arbitrary
quantization formats, or reduced-precision activations.

The synchronous API requires `pyodide.ffi.can_run_sync()` and a stack-switching
Python entrypoint: `runPythonAsync()` or PyProxy `callPromising()`. Code entered
through `runPython()`, a direct synchronous PyProxy call, or a runtime without
JSPI must use `init_async`, `synchronize_async`, and `to_cpu_async`. This is a
`PrivateUse1` backend named `webgpu`;
it does not claim CUDA compatibility and `torch.cuda.is_available()` remains
false.

The development backend compiles against the exact Dawn-style headers shipped
with Emdawnwebgpu and reuses pinned torch-webgpu C++ dispatch helpers and WGSL.
A compute-only C API profile carries browser calls as side-module `EM_JS`
because full Emdawn JavaScript must be final-linked into a main module and is
absent from stock Pyodide. See the
[architecture](docs/webgpu-browser-architecture.md),
[operator support table](docs/webgpu-operator-support.md), and
[third-party notices](THIRD_PARTY_NOTICES.md).

## What works

| Area | Current status |
| --- | --- |
| Tensor creation, indexing, shapes, dtypes, and NumPy interop | Supported by smoke tests and the selected upstream suite |
| Autograd, `torch.nn`, and optimizers | Supported by runtime smoke tests |
| `torch.linalg` | LAPACK-backed; 71 selected upstream linalg tests pass |
| Serialization and selected `torch.func` operations | Supported by runtime smoke tests |
| Experimental WebGPU (`r7`) | `float32` eager inference subset with JSPI-backed synchronous Python APIs; see the operator table |
| CUDA, ROCm, MPS, or XPU | Not available |
| Multiprocessing, distributed training, and shared-memory tensors | Not available |
| `torch.compile`, C++ extensions, and multithreaded CPU execution | Not available |

The table describes tested scope, not complete API compatibility. See
[CAVEATS.md](CAVEATS.md) for behavioral constraints and
[the upstream test policy](docs/upstream-tests.md) for the exact test
inventory.

## How releases are validated

Every release is built from a pinned PyTorch commit and an ordered patch
series. CI then:

1. validates the wheel metadata, WebAssembly structure, dynamic dependencies,
   and absence of shared memory and atomics;
2. imports the wheel in the pinned Pyodide runtime and exercises tensor
   operations, autograd, `torch.nn`, optimization, serialization, `torch.func`,
   and LAPACK-backed linear algebra;
3. runs 654 selected upstream PyTorch CPU tests with zero runtime skips,
   expected failures, failures, or errors;
4. runs real eager kernels, copies/readback, composed rotary encoding, and a
   tiny decoder block in Chromium with SwiftShader WebGPU, requiring zero
   validation errors or implicit CPU fallbacks;
5. publishes a SHA-256 digest, a machine-readable build manifest, and a GitHub
   artifact attestation;
6. automatically promotes only that successful build to an immutable release,
   then browser-smokes and deploys Pages from the Publisher run's exact
   artifact.

The selected suite is deliberately auditable. All 16 generated-test exclusions
and every probed test not admitted to CI are documented in
[`docs/upstream-tests.md`](docs/upstream-tests.md).

## Documentation

| Document | Contents |
| --- | --- |
| [Caveats](CAVEATS.md) | Runtime limitations, unsupported APIs, performance, and ABI constraints |
| [Compatibility](docs/compatibility.md) | Exact source, toolchain, Python, Pyodide, and WebAssembly pins |
| [Upstream test policy](docs/upstream-tests.md) | Passing tests, explicit exclusions, and collection accommodations |
| [Build and release](docs/building.md) | Build pipeline, validation, caching, and version updates |
| [Browser WebGPU architecture](docs/webgpu-browser-architecture.md) | Side-module Emdawn profile, initialization, handles, lifetimes, and memory |
| [Browser WebGPU operator support](docs/webgpu-operator-support.md) | Verified, compile-only, missing-kernel, fallback, and synchronous-readback status |
| [Transformers/Qwen target](docs/transformers-qwen.md) | Blockers and hermetic acceptance plan for a real tiny Qwen2 model |
| [Contributing](CONTRIBUTING.md) | Development workflow and pull-request expectations |
| [Security policy](SECURITY.md) | Vulnerability reporting and release verification |

## Roadmap

- Build and publish a tested PyTorch × Pyodide compatibility matrix.
- Expand the pinned upstream CPU/WebAssembly test suite, especially autograd
  and `torch.nn`.
- Expand the experimental WebGPU operator and dtype coverage while preserving
  explicit unsupported-operation errors and zero implicit CPU fallback.
- Pass a hermetic tiny Qwen2 Transformers forward test before pursuing
  cache-backed generation or public-size checkpoints.

## License

The build scripts, patches, tests, and playground code in this repository are
available under the [MIT License](LICENSE). PyTorch remains under its upstream
BSD-3-Clause license, and bundled dependencies retain their respective
licenses. The experimental WebGPU patch includes Apache-2.0-licensed portions
adapted from `torch-webgpu`; see [Third-party notices](THIRD_PARTY_NOTICES.md).
PyTorch names and trademarks belong to their respective owners.
