# Vendored WebGPU sources

These snapshots are immutable build inputs. `scripts/stage_webgpu_sources.py`
copies them into the pinned PyTorch checkout; normal wheel builds do not fetch
them from the network.

| Directory | Exact source | Local treatment |
| --- | --- | --- |
| `torch-webgpu` | `https://github.com/jmaczan/torch-webgpu.git` at `a4369ff0f61f4e58cbffb048cee85047b33dacba` | Unmodified `csrc`, `LICENSE`, and `README.md`, plus a local `COMMIT` pin marker; CMake compiles an explicit allowlist. |
| `emdawnwebgpu` | Dawn release package `emdawnwebgpu_pkg-v20251002.162335.zip`, Dawn `01940842b667a7812d0e4ca0ef4367fbec294241` | Unmodified package; build consumes generated C/C++ headers. Archive SHA-512 is in `config/build.toml`. |

Browser-specific adaptations live in the PyTorch patch, not in these source
trees. Complete attribution and license mapping is in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
