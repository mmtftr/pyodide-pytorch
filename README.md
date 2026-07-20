# PyTorch for Pyodide

Reproducible, unofficial, CPU-only PyTorch wheels for Pyodide. The build is
deliberately single-threaded: it does not enable Emscripten pthreads or
WebAssembly shared memory, and ATen executes intra-op and inter-op work inline.

> [!IMPORTANT]
> The pinned upgrade target is PyTorch `cf30153` (`2.13.0`) on Pyodide
> `314.0.2`, CPython `3.14.2`, and Emscripten `5.0.3`.
> Release assets are published only after the canonical CI full build, binary
> validation, and Pyodide smoke test pass for the exact source commit. Wheels
> are ABI-specific and must be used with the exact Pyodide version named by the
> release.

## What this repository provides

- One reviewed manifest, [`config/build.toml`](config/build.toml), pinning every
  ABI-relevant input.
- Small, ordered `git apply` patches instead of a long-lived PyTorch fork.
- A native `protoc` bootstrap built from PyTorch's own pinned protobuf
  submodule, so cross-compilation never executes a target Wasm binary.
- Static libtorch linking into a single Pyodide side module.
- Deterministic post-processing that removes headers, static archives, command
  line tools, and other build-only payloads.
- Binary validation of the wheel tag, RECORD hashes, WebAssembly magic,
  target features, non-shared memory, dynamic dependencies, and unresolved
  PyTorch-owned symbols.
- A real Pyodide/Node smoke test covering tensor operations, autograd,
  `torch.nn`, an optimizer step, serialization, and `torch.func`.
- Release SHA-256 files and a machine-readable input manifest. Public
  repositories also receive GitHub artifact attestations. Generated wheels are
  release assets, never Git objects.

The earlier wheel shared in
[`mat3ra/api-examples#286`](https://github.com/mat3ra/api-examples/pull/286)
was useful as a behavioral reference, but it has no reproducible build recipe.
This repository does not copy or republish that binary.

## Threading model

Turning off OpenMP and optimized CPU backends is not sufficient by itself:
PyTorch's native ATen schedulers can still construct `std::thread` pools. The
patch set therefore makes the constraint explicit:

| Layer | Enforcement |
| --- | --- |
| Build | No `-pthread`; OpenMP, NNPACK/QNNPACK/XNNPACK, FBGEMM, MKL, MKLDNN, distributed, and accelerator backends are off. |
| ATen | `get_num_threads()` and `get_num_interop_threads()` return `1`; attempts to set either to another value fail. |
| Scheduling | Intra-op and inter-op callbacks execute synchronously on the current thread. |
| Autograd | CPU autograd stays on the calling thread, including re-entrant execution. |
| Artifact | Validation rejects shared Wasm memory and the `atomics` target feature. |

Some libc and C++ standard-library pthread symbol names can remain in dead or
stubbed code. The meaningful Wasm invariant is that the module neither imports
shared memory nor opts into atomics, and no runtime path creates a worker.
Filesystem-backed multiprocessing storage is explicitly unsupported: its
fork/socket manager is omitted, while unrelated storage weak-reference APIs
remain available.

## Use a release wheel

Host the wheel on a CORS-enabled origin, then load it into the matching
Pyodide runtime. `loadPackage()` does not resolve arbitrary wheel dependencies,
so load the packages explicitly:

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
    print(torch.__version__, x.grad)
  `);
</script>
```

Verify the release checksum before deploying the wheel. The wheel uses
Pyodide's standardized `pyemscripten_2026_0_wasm32` platform tag. This
repository publishes it through GitHub Releases; a CORS-enabled static origin
also works.

## Build

The GitHub Actions workflow is the canonical build environment. Run **Build
PyTorch wheel** with `workflow_dispatch`, or push a change to an ABI-relevant
path on `main`. The accepted release name is pinned in `config/build.toml`; a
matching tag builds, tests, attests when supported, and publishes the assets.
Automation that cannot create tags may instead create `release/<release-tag>`
directly at the current `main` commit. After the same build and tests pass, the
workflow creates the corresponding tag and release. This branch trigger is
one-shot: updating an existing release branch never republishes it.

GitHub does not support artifact attestations for user-owned private
repositories. In that case the attestation step is skipped; checksum, binary,
and runtime validation still gate publication. If a release build uploads a
verified artifact but a later publication step fails, run **Publish verified
release artifact** with the source run ID. Automation without workflow-dispatch
access may create `publish/<run-id>` at current `main`. The recovery workflow
checks the source run and every critical build step, revalidates the artifact
and manifest against the current pins, reruns the Pyodide smoke test, and
refuses to overwrite an existing release. A retry can use
`publish/<run-id>-retry-<number>`; each run reports its URL through the
`pytorch-pyodide/release` commit status.

The pipeline performs these steps:

1. Checks out the exact PyTorch commit and its recursively pinned submodules.
2. Verifies every patch with `git apply --check`, then applies it in lexical
   order.
3. Builds a host `protoc` from the matching protobuf submodule.
4. Installs the pinned `pyodide-build` and Emscripten toolchains.
5. Runs `pyodide build --skip-emscripten-install --exports=whole_archive` with
   the CPU-only feature set and the separately pinned Emscripten installation.
6. Prunes build-time payloads and rewrites wheel RECORD hashes
   deterministically.
7. Rejects an ABI mismatch, native/non-Wasm `.so`, static archive, shared
   memory, atomics feature, missing dependency, unresolved project symbol, or
   oversized wheel.
8. Imports the wheel in the pinned Pyodide runtime and runs the smoke suite.

The workflow keeps `PYODIDE_XBUILDENV_PATH` stable and caches the cross-build
environment by Python, Pyodide, and `pyodide-build` version. Emscripten, native
`protoc`, and compiler outputs use separate ABI-scoped caches. Runtime tests use
Node.js 24, matching the current `pyodide-build` recommendation.

Fast local repository checks do not require a PyTorch checkout:

```bash
python3 scripts/config.py check
python3 scripts/validate_patches.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash -n scripts/*.sh
```

The validation workflow additionally checks out the exact pinned PyTorch
commit without submodules and applies the complete patch series. This is a
fast drift check only; it does not substitute for the full WebAssembly build
and runtime smoke test.

Cross-compiling PyTorch is resource-intensive. For a local full build, mirror
the commands in [`.github/workflows/build.yml`](.github/workflows/build.yml) on
Linux with at least 16 GiB RAM and ample free disk.

## Updating versions

Do not change a wheel filename to claim compatibility with another Pyodide
release. Instead:

1. Add or update the complete pins in `config/build.toml`.
2. Rebase the patch series on the new PyTorch commit and regenerate focused
   patches with `git format-patch`.
3. Let `git apply --check` catch upstream drift.
4. Build a fresh wheel and pass both binary validation and the runtime suite.
5. Add tests for any newly supported surface before publishing.

Pyodide's platform ABI is versioned independently of its raw Emscripten
version. Keep the runtime, cross-build environment, Python tag, platform tag,
and Emscripten pin together, then require a fresh runtime smoke test. See
[compatibility and scope](docs/compatibility.md) for intentional omissions.

## Project status

This is an experimental downstream build, not an official PyTorch or Pyodide
distribution. The PyTorch name and trademarks belong to their respective
owners. Build scripts in this repository are MIT-licensed; PyTorch remains
under its upstream BSD-3-Clause license.

The upstream tracking discussion is
[`pyodide/pyodide-recipes#193`](https://github.com/pyodide/pyodide-recipes/issues/193).
