# PyTorch for Pyodide

[![Validate repository](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/validate.yml)
[![Build PyTorch wheel](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/build.yml)
[![Deploy playground](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/mmtftr/pyodide-pytorch/actions/workflows/pages.yml)
[![Latest release](https://img.shields.io/github/v/release/mmtftr/pyodide-pytorch?display_name=tag&sort=date)](https://github.com/mmtftr/pyodide-pytorch/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run PyTorch in a browser or another Pyodide environment. This repository
produces a reproducible, CPU-only WebAssembly wheel, tests it inside the exact
Pyodide runtime it targets, and publishes the wheel with checksums and build
provenance.

**[Try the browser playground](https://mmtftr.github.io/pyodide-pytorch/)**
· [Download the latest release](https://github.com/mmtftr/pyodide-pytorch/releases/latest)
· [Read the caveats](CAVEATS.md)

> [!IMPORTANT]
> This is an experimental downstream distribution. It is not an official
> PyTorch or Pyodide release.

## Current release

| Component | Version |
| --- | --- |
| PyTorch | `2.13.0+pyodide314.0.2` |
| Pyodide | `314.0.2` |
| Python | `3.14.2` (`cp314`) |
| WebAssembly platform | `pyemscripten_2026_0_wasm32` |
| Release | `torch-2.13.0-pyodide-314.0.2-r2` |

The complete, ABI-relevant configuration lives in
[`config/build.toml`](config/build.toml). A wheel is compatible only with the
Pyodide ABI recorded in its release manifest; it is not a general-purpose
CPython wheel.

## Try it

The [playground](https://mmtftr.github.io/pyodide-pytorch/) loads the latest
verified release in a Web Worker and provides:

- a CodeMirror Python editor and separate output console;
- autocompletion with `Tab` or `Ctrl+Space`;
- runtime-derived signatures and documentation on hover;
- examples for tensors, autograd, neural networks, optimization, and
  `torch.linalg`;
- restart, cancellation, and versioned browser caching.

The first load downloads Pyodide, its Python dependencies, and the PyTorch
wheel. Compatible subsequent visits reuse the versioned runtime cache.

## Use it in a web application

The wheel is not published on PyPI. Load the matching Pyodide runtime and the
verified wheel deployed with the playground:

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

## What works

| Area | Current status |
| --- | --- |
| Tensor creation, indexing, shapes, dtypes, and NumPy interop | Supported by smoke tests and the selected upstream suite |
| Autograd, `torch.nn`, and optimizers | Supported by runtime smoke tests |
| `torch.linalg` | LAPACK-backed; 71 selected upstream linalg tests pass |
| Serialization and selected `torch.func` operations | Supported by runtime smoke tests |
| CUDA, ROCm, MPS, XPU, or WebGPU | Not available |
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
4. publishes a SHA-256 digest, a machine-readable build manifest, and a GitHub
   artifact attestation.

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
| [Contributing](CONTRIBUTING.md) | Development workflow and pull-request expectations |
| [Security policy](SECURITY.md) | Vulnerability reporting and release verification |

## Roadmap

- Build and publish a tested PyTorch × Pyodide compatibility matrix.
- Expand the pinned upstream CPU/WebAssembly test suite, especially autograd
  and `torch.nn`.
- Investigate WebGPU support. This requires substantial PyTorch backend and
  browser integration work and is not currently promised.

## License

The build scripts, patches, tests, and playground code in this repository are
available under the [MIT License](LICENSE). PyTorch remains under its upstream
BSD-3-Clause license, and bundled dependencies retain their respective
licenses. PyTorch names and trademarks belong to their respective owners.
