# Build and release

This document describes the repository's implementation and maintainer
workflow. Start with the [README](../README.md) if you only want to use the
wheel.

## Sources of truth

The build is defined by:

1. [`config/build.toml`](../config/build.toml), which pins the PyTorch source,
   Pyodide ABI, Python, Emscripten, LAPACK package, host tools, and release tag;
2. [`config/build-constraints.txt`](../config/build-constraints.txt), which
   constrains Python build dependencies;
3. the ordered patches under [`patches/pytorch/`](../patches/pytorch/);
4. the build, post-processing, and validation scripts under
   [`scripts/`](../scripts/).

Generated wheels are release artifacts and must not be committed.

## Build pipeline

The canonical GitHub Actions build performs the following operations:

1. Read and validate `config/build.toml`.
2. Check out the pinned PyTorch commit and recursive submodules.
3. Check every patch with `git apply --check`, then apply patches in filename
   order.
4. Build a native `protoc` from PyTorch's pinned protobuf submodule so the
   cross-build never executes a WebAssembly target binary on the host.
5. Install the pinned Emscripten and `pyodide-build` toolchains.
6. Run `pyodide build --skip-emscripten-install --exports=whole_archive` with
   the CPU-only feature configuration.
7. Remove headers, static archives, command-line programs, and other build-only
   payloads from the raw wheel.
8. Run `pyodide auditwheel repair` to vendor the pinned Pyodide
   `libopenblas.so` and set the WebAssembly runtime search path.
9. Rewrite wheel `RECORD` hashes and archive entries deterministically.
10. Validate the binary and execute the smoke and selected upstream tests in
    the pinned Pyodide runtime.

The canonical implementation is
[`.github/workflows/build.yml`](../.github/workflows/build.yml).

## Patch series

The patch series in [`patches/pytorch/`](../patches/pytorch/) contains the
Pyodide cross-build changes, single-threaded ATen and autograd behavior, static
linking changes, unsupported multiprocessing exclusions, the LAPACK ABI
adaptation, and Emscripten fixes required by the pinned PyTorch revision.

Patches must remain focused and ordered. A generally useful Emscripten fix
should be proposed upstream to PyTorch; a release-specific compatibility patch
can remain here while its upstream status is documented.

## Binary validation

[`scripts/validate_wheel.py`](../scripts/validate_wheel.py) checks:

- Python and Pyodide platform tags;
- wheel metadata and `RECORD` hashes;
- WebAssembly magic and dynamic-library structure;
- required runtime files;
- absence of static archives and other build-only files;
- absence of shared memory and the atomics target feature;
- the exact `libopenblas.so` dependency and its vendored runtime path;
- unresolved PyTorch-owned symbols;
- the configured maximum wheel size.

[`scripts/verify_release_artifact.py`](../scripts/verify_release_artifact.py)
verifies the wheel digest, manifest, builder commit, configuration, patches,
and build-script inputs before a release artifact is published or deployed to
the playground.

## Runtime validation

[`tests/smoke.mjs`](../tests/smoke.mjs) loads the wheel into the pinned Pyodide
runtime and covers:

- tensor creation and arithmetic;
- autograd;
- LAPACK-backed inverse, solve, eigenvalue, and Cholesky operations;
- `torch.nn` and an optimizer step;
- serialization;
- selected `torch.func` operations;
- version, platform, and single-thread invariants.

[`tests/upstream.mjs`](../tests/upstream.mjs) then copies selected test modules
verbatim from the exact pinned PyTorch checkout and executes the manifest in
[`tests/upstream_cpu_wasm.json`](../tests/upstream_cpu_wasm.json). The current
contract is 654 passing tests with no runtime skips, expected failures,
failures, errors, or unexpected successes.

See [`docs/upstream-tests.md`](upstream-tests.md) for the complete selection and
exclusion policy.

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

Given a built wheel and a checkout at the pinned PyTorch commit, reproduce the
upstream runtime gate with:

```bash
npm install --no-save \
  "pyodide@$(python3 scripts/config.py get PYODIDE_VERSION)"
node tests/upstream.mjs \
  path/to/torch.whl \
  path/to/pytorch \
  "$(python3 scripts/config.py get PYTORCH_VERSION)" \
  "$(python3 scripts/config.py get PYTORCH_REF)"
```

Pass `--list-tests` to collect and print the generated upstream test IDs without
executing them. The runner rejects a source checkout, manifest, PyTorch
version, or wheel commit that does not match the pinned configuration.

## Full local build

The full WebAssembly build is resource-intensive. Reproduce the pinned
environment and commands from
[`.github/workflows/build.yml`](../.github/workflows/build.yml) on Linux with
at least 16 GiB RAM and sufficient disk space.

[`scripts/build_wheel.sh`](../scripts/build_wheel.sh) is the build entry point,
but it assumes that the host compiler, Emscripten, Pyodide cross-build
environment, native `protoc`, patched PyTorch checkout, and LAPACK side module
have already been prepared.

Before CMake configuration, the build script runs
[`scripts/stage_webgpu_sources.py`](../scripts/stage_webgpu_sources.py). It
copies the checked-in, pinned torch-webgpu and Emdawnwebgpu snapshots into the
PyTorch `third_party` directory. This step performs no network access. Vendor
tree hashes and the staging-script hash are included in the release manifest.

## Build cache

GitHub Actions uses separate caches for:

- the Pyodide cross-build environment;
- Emscripten;
- the native protobuf compiler;
- the pinned Pyodide `libopenblas` package archive;
- `ccache` compiler output;
- playground npm packages.

ABI-specific keys include the relevant PyTorch, Pyodide, Python, Emscripten,
build-tool, configuration, patch, and build-script inputs. Compiler-cache
restore keys omit the workflow run identifier so a compatible previous build
can be reused; the saved key remains unique per run attempt.

The playground separately uses a service worker and the browser Cache Storage
API. Application metadata uses a network-first policy. Immutable Pyodide
assets, Python packages, and versioned wheel files use a cache-first policy.
Frontend assets have an explicit version in `site/app.js`; increment it when a
deployment must invalidate the application shell.

## Release artifacts

Each GitHub release contains:

- the PyTorch wheel;
- a SHA-256 file for the wheel;
- `build-manifest.json`, including the source commit, full build
  configuration, input hashes, wheel filename, size, and digest;
- a GitHub artifact attestation when supported by the repository
  configuration.

The release workflow revalidates downloaded artifacts before publication and
does not overwrite an existing release.

The Pages workflow downloads the current release, verifies it with
[`scripts/verify_release_artifact.py`](../scripts/verify_release_artifact.py),
and deploys the wheel beside the playground. Browser requests therefore use a
same-origin wheel URL.

## Updating versions

Never rename a wheel to claim compatibility with another Pyodide version. For a
version update:

1. update all related pins in `config/build.toml`;
2. rebase or regenerate the patch series against the new PyTorch commit;
3. update `config/build-constraints.txt` when host-tool pins change;
4. run the fast checks and patch-applicability job;
5. produce a new wheel with the full build workflow;
6. pass binary validation, smoke tests, and the selected upstream suite;
7. publish a new release tag and manifest;
8. update the version table in the README and compatibility documentation.

The runtime, cross-build environment, Python tag, platform tag, and Emscripten
pin must be updated and tested as one ABI tuple.

## Repository layout

| Path | Purpose |
| --- | --- |
| `config/build.toml` | Build, ABI, toolchain, and release pins |
| `config/build-constraints.txt` | Python build-dependency constraints |
| `patches/pytorch/` | Ordered patches applied to the pinned PyTorch source |
| `vendor/` | Pinned torch-webgpu and Emdawnwebgpu source snapshots |
| `scripts/stage_webgpu_sources.py` | Deterministic, network-free vendor staging |
| `scripts/build_wheel.sh` | PyTorch and `pyodide-build` entry point |
| `scripts/fetch_lapack.py` | Pinned Pyodide LAPACK download and verification |
| `scripts/postprocess_wheel.py` | Deterministic pruning and repacking |
| `scripts/validate_wheel.py` | Wheel and WebAssembly validation |
| `scripts/verify_release_artifact.py` | Release digest, manifest, and input verification |
| `tests/` | Repository, smoke, and selected upstream tests |
| `docs/` | Compatibility, testing, and maintainer documentation |
| `site/` | Browser playground source |
| `.github/workflows/` | Validation, build, release, and Pages pipelines |
