# PyTorch for Pyodide

This repository builds an unofficial, CPU-only PyTorch wheel for Pyodide. The
wheel is compiled as WebAssembly, uses Pyodide's platform ABI, and is validated
inside the matching Pyodide runtime before release.

## Current build target

| Component | Version |
| --- | --- |
| PyTorch | `cf30153c4c131c8164ee7798e5022d810682e2cb` (`2.13.0`) |
| Wheel version | `2.13.0+pyodide314.0.2` |
| Pyodide runtime | `314.0.2` |
| `pyodide-build` | `0.36.0` |
| Python | `3.14.2` / `cp314` |
| Emscripten | `5.0.3` |
| Platform tag | `pyemscripten_2026_0_wasm32` |
| LAPACK | Pyodide `libopenblas` `0.3.28` |
| Release | `torch-2.13.0-pyodide-314.0.2-r2` |

All ABI-relevant versions are defined in [`config/build.toml`](config/build.toml).
The wheel must be loaded with the Pyodide version recorded in its release
manifest.

## Supported configuration

The current build has the following constraints:

- CPU execution only.
- One intra-op thread and one inter-op thread.
- No Emscripten pthreads, WebAssembly shared memory, or atomics target feature.
- No OpenMP, MKL, MKLDNN, FBGEMM, XNNPACK, QNNPACK, NNPACK, distributed, CUDA,
  ROCm, or other accelerator backends.
- Intra-op, inter-op, and CPU autograd callbacks execute synchronously on the
  Pyodide worker thread.
- Filesystem-backed multiprocessing storage is not included.
- Eigen provides the CPU BLAS implementation. The wheel vendors the matching
  Pyodide `libopenblas.so` side module for LAPACK-backed `torch.linalg`
  operations.

The patch set keeps the Python bindings and statically links the required
libtorch components into the `torch._C` Pyodide side module. `torch._C` has one
dynamic dependency, the vendored `torch.libs/libopenblas.so` LAPACK module.

## Release artifacts

Each release contains:

- The PyTorch wheel.
- A SHA-256 file for the wheel.
- `build-manifest.json`, containing the build configuration, source commit,
  input hashes, wheel filename, wheel size, and wheel digest.
- A GitHub artifact attestation when the repository configuration supports it.

The release workflow revalidates downloaded artifacts before publication and
does not overwrite existing releases.

## Loading the wheel

Host the wheel on an origin that allows browser requests, load the dependencies
that Pyodide already packages, install `filelock`, and then load the wheel:

```html
<script type="module">
  import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.2/full/pyodide.mjs";

  const pyodide = await loadPyodide();

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
    await micropip.install("filelock")
  `);

  await pyodide.loadPackage("https://your-origin.example/torch-...whl");

  await pyodide.runPythonAsync(`
    import torch

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    x.square().sum().backward()
    print(torch.__version__)
    print(x.grad)
  `);
</script>
```

Verify the wheel against the published SHA-256 value before deploying it to a
static origin.

## Browser playground

The [browser playground](https://mmtftr.github.io/pyodide-pytorch/) contains a
CodeMirror Python editor and a separate output console. It supports Python
syntax highlighting, editor key bindings, runtime-backed completion, example
programs, code reset and copy controls, runtime restart, and cancellation by
terminating the worker.

Editor controls:

| Input | Action |
| --- | --- |
| `Ctrl+Space` | Open completion suggestions |
| `Tab` | Accept the active completion, or indent when completion is closed |
| `Ctrl+Enter` or `Shift+Enter` | Run the current editor contents |
| Hover a Python or `torch` symbol | Inspect its runtime signature and docstring |

Completion and hover information are resolved in the loaded Pyodide worker.
The worker uses Python's `inspect` module against builtins, imported modules,
PyTorch namespaces, and objects created by executed code. Inspection requests
accept dotted Python identifiers only, return bounded documentation text, and
are skipped while user code is running. Results are cached in memory until the
runtime or user namespace changes. Hover tooltips link to the corresponding
Python or PyTorch reference page when a stable reference URL is available.

The playground implementation is split into:

- `site/app.js`: editor state, controls, release-manifest loading, and worker
  message handling.
- `site/worker.js`: Pyodide initialization, package loading, runtime validation,
  Python execution, stdout and stderr forwarding, completion, and symbol
  inspection.
- `site/service-worker.js`: Cache Storage policies for the application shell,
  Pyodide distribution files, Python packages, and versioned PyTorch wheel.
- `site/index.html` and `site/styles.css`: application layout and styling.

The Pages workflow downloads the current release artifacts, verifies them with
[`scripts/verify_release_artifact.py`](scripts/verify_release_artifact.py), and
places the wheel beside the static application. The browser therefore loads the
wheel from the same origin as the playground. Publishing or recovering a
release triggers a new Pages deployment.

## Build implementation

The build performs these operations:

1. Read and validate `config/build.toml`.
2. Check out the pinned PyTorch commit and recursive submodules.
3. Check every patch with `git apply --check` and apply the patches in filename
   order.
4. Build a native `protoc` from PyTorch's pinned protobuf submodule so the
   cross-build never attempts to execute a WebAssembly target binary.
5. Install the pinned Emscripten and `pyodide-build` toolchains.
6. Run `pyodide build --skip-emscripten-install --exports=whole_archive` with
   the configured CPU-only feature flags.
7. Remove headers, static archives, command-line programs, and other build-only
   payloads from the raw wheel.
8. Use `pyodide auditwheel repair` to vendor the pinned Pyodide
   `libopenblas.so` and set the WebAssembly runtime search path.
9. Rewrite wheel `RECORD` hashes and produce deterministic archive metadata.
10. Validate the wheel and run the Pyodide runtime tests.

The patch series is stored in `patches/pytorch/`. It contains the Pyodide
cross-build changes, single-threaded ATen and autograd implementation, static
linking changes, unsupported multiprocessing exclusions, and Emscripten build
fixes required by the pinned source revision.

## Validation

[`scripts/validate_wheel.py`](scripts/validate_wheel.py) checks:

- The Python and Pyodide platform tags.
- Wheel metadata and `RECORD` hashes.
- WebAssembly magic and dynamic-library structure.
- Required runtime files.
- Absence of static archives and other build-only files.
- Absence of shared memory and the atomics target feature.
- The exact `libopenblas.so` dynamic dependency and its vendored runtime path.
- Unresolved PyTorch-owned symbols.
- The configured maximum wheel size.

[`tests/smoke.mjs`](tests/smoke.mjs) imports the wheel in the pinned Pyodide
runtime and covers tensor operations, autograd, LAPACK-backed inverse, solve,
eigenvalue, and Cholesky operations, `torch.nn`, an optimizer step,
serialization, and `torch.func`.

Fast repository validation checks configuration parsing, patch syntax, helper
scripts, artifact verification, packaging, and WebAssembly inspection. Patch
applicability is also tested against the exact pinned PyTorch commit without
running the full build.

## Build cache

GitHub Actions uses separate caches for:

- The Pyodide cross-build environment.
- Emscripten.
- The native protobuf compiler.
- The pinned Pyodide `libopenblas` package archive.
- `ccache` compiler output.
- Playground npm packages.

The browser playground also uses a service worker and the Cache Storage API.
HTML, the release manifest, and application files use a network-first policy so
deployments update normally. Versioned Pyodide files, Python wheels, and the
versioned PyTorch wheel use a cache-first policy. This avoids downloading the
runtime and 25 MB PyTorch wheel again on a compatible subsequent visit. The
**Clear cache** control removes only caches owned by this playground. Browser
storage quotas and eviction policies still apply.

The HTML entry point uses an explicit frontend asset version for the bundled
editor, worker, and service worker. Change that version and the shell cache name
when those files require a cache-breaking deployment. Runtime wheels do not
need this manual version because their release filenames are immutable.

The ABI-specific cache keys include the relevant PyTorch, Pyodide, Python,
Emscripten, build-tool, configuration, patch, and build-script inputs. Compiler
cache restore keys omit the workflow run identifier so a compatible previous
build can be reused; the saved key remains unique per run attempt.

## Repository layout

| Path | Purpose |
| --- | --- |
| `config/build.toml` | Build, ABI, toolchain, and release pins |
| `config/build-constraints.txt` | Python build dependency constraints |
| `patches/pytorch/` | Ordered patches applied to the pinned PyTorch source |
| `scripts/build_wheel.sh` | PyTorch and `pyodide-build` entry point |
| `scripts/fetch_lapack.py` | Pinned Pyodide LAPACK download and verification |
| `scripts/postprocess_wheel.py` | Deterministic pruning and repacking |
| `scripts/validate_wheel.py` | Wheel and WebAssembly validation |
| `scripts/verify_release_artifact.py` | Release checksum, manifest, and input verification |
| `tests/` | Repository and Pyodide runtime tests |
| `site/` | Browser playground source |
| `.github/workflows/` | Validation, build, release, and Pages pipelines |

## Local checks

The fast checks do not require a PyTorch checkout:

```bash
python3 scripts/config.py check
python3 scripts/validate_patches.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash -n scripts/*.sh

npm ci --prefix site
npm run build --prefix site
```

The full WebAssembly build is resource-intensive. Use the commands and pinned
environment in [`.github/workflows/build.yml`](.github/workflows/build.yml) on
Linux with at least 16 GiB RAM and sufficient disk space.

## Updating versions

Do not rename a wheel to claim compatibility with a different Pyodide version.
For a version update:

1. Update all related pins in `config/build.toml`.
2. Rebase and regenerate the patch series against the new PyTorch commit.
3. Run the patch-applicability checks.
4. Produce a new wheel with the full build workflow.
5. Pass binary validation and the Pyodide runtime suite.
6. Publish a new release tag and manifest.

Pyodide's platform ABI is versioned independently of the Emscripten version.
The runtime, cross-build environment, Python tag, platform tag, and Emscripten
pin must be updated and tested together. See
[`docs/compatibility.md`](docs/compatibility.md) for the current implementation
scope.

## Future work

- Build and publish a tested compatibility matrix across multiple Pyodide and
  PyTorch versions.
- Run a selected subset of the upstream PyTorch test suite in Pyodide in
  addition to the repository smoke tests.
- Investigate WebGPU support. This is exploratory: the effort depends on the
  amount of PyTorch backend and browser integration code that must be
  implemented or adapted.

## Status and licensing

This is an experimental downstream build and is not an official PyTorch or
Pyodide distribution. Build and support status is tracked in this repository.
The related upstream packaging discussion is
[`pyodide/pyodide-recipes#193`](https://github.com/pyodide/pyodide-recipes/issues/193).

Repository build scripts are MIT-licensed. PyTorch remains under its upstream
BSD-3-Clause license. PyTorch names and trademarks belong to their respective
owners.
