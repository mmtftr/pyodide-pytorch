# Contributing

The source of truth for a build is `config/build.toml`, followed by the ordered
patches under `patches/pytorch/`. Generated wheels are release assets and must
not be committed.

Before proposing a version bump:

1. Pin the full PyTorch commit, Pyodide version, Python patch version, and
   Emscripten version in `config/build.toml`.
2. Rebase every patch with `git am` or regenerate it with `git format-patch`.
3. Run `python scripts/config.py check` and `python -m unittest discover`.
4. Dispatch the build workflow and attach its smoke-test result to the PR.

Keep patches focused and include the upstream rationale in their commit
message. Prefer upstreaming generally useful Emscripten changes to PyTorch.
