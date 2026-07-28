# Contributing

Contributions are welcome for build portability, WebAssembly compatibility,
test coverage, documentation, and the browser playground. This project carries
a large downstream patch set, so changes must remain reproducible against the
exact versions in [`config/build.toml`](config/build.toml).

## Before opening a pull request

1. Search existing issues and pull requests for overlapping work.
2. Keep the change focused; separate version bumps, patch rebases, test-suite
   expansion, and playground features when they can be reviewed independently.
3. Add or update tests for behavioral changes.
4. Document user-visible limitations in [CAVEATS.md](CAVEATS.md).
5. Run the relevant local checks below.

Bug reports should use the build-failure issue template and include the exact
release or workflow run, browser or Node version, Pyodide version, minimal
reproduction, and complete error output.

## Fast checks

These checks do not require a PyTorch checkout:

```bash
python3 scripts/config.py check
python3 scripts/validate_patches.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash -n scripts/*.sh

npm ci --prefix site
npm run build --prefix site
```

Run `git diff --check` before committing.

## PyTorch patches

Patches live under [`patches/pytorch/`](patches/pytorch/) and are applied in
filename order.

- Rebase a patch with `git am` or regenerate it with `git format-patch`.
- Keep each patch focused and explain the upstream constraint in its commit
  message.
- Verify the complete series with `python3 scripts/validate_patches.py`.
- Prefer contributing generally useful Emscripten fixes to PyTorch upstream.
- Document whether an upstream pull request exists and when the downstream
  patch can be removed.

Do not modify copied upstream tests to make them pass. The runner copies test
files verbatim from the pinned PyTorch checkout.

## Expanding the upstream test gate

[`tests/upstream_cpu_wasm.json`](tests/upstream_cpu_wasm.json) is the executable
test-selection manifest. Follow
[`docs/upstream-tests.md`](docs/upstream-tests.md) when adding tests.

Every selected test must:

- come from the exact pinned PyTorch source;
- pass in the matching Pyodide runtime;
- have a stable generated test ID;
- run within the build workflow's practical browser resource limits.

Whole-module selections require every generated exclusion to have a nonempty,
specific reason. Tests probed but not admitted to CI must not be represented as
passing.

## Version updates

A version bump is an ABI change, not a metadata-only edit:

1. Pin the full PyTorch commit, Pyodide version, Python patch version,
   `pyodide-build`, platform tag, Emscripten, LAPACK package, and host tools in
   `config/build.toml`.
2. Keep `config/build-constraints.txt` synchronized with the host-tool pins.
3. Rebase or regenerate every PyTorch patch.
4. Pass repository validation and patch applicability.
5. Dispatch the full build and attach the successful workflow run to the pull
   request.
6. Update the README and compatibility table only after the new wheel passes
   binary, smoke, and upstream runtime validation.

See [`docs/building.md`](docs/building.md) for the canonical pipeline.

## Pull requests

A pull request should explain:

- what changed and why;
- the user or maintainer impact;
- which ABI or source pins are affected;
- tests and workflow runs used for validation;
- new limitations, exclusions, or follow-up work.

Do not commit generated wheels, downloaded PyTorch sources, build directories,
compiler caches, or playground bundles.

By contributing, you agree that your changes are provided under this
repository's [MIT License](LICENSE).
