# Third-party notices

## torch-webgpu

The experimental browser WebGPU backend in
`vendor/torch-webgpu` is a source snapshot of
[jmaczan/torch-webgpu](https://github.com/jmaczan/torch-webgpu) commit
`a4369ff0f61f4e58cbffb048cee85047b33dacba`.

The browser backend compiles its binary/unary dispatch implementations and
WGSL and adapts its matrix multiplication and softmax implementations. Those
portions are licensed under the Apache License, Version 2.0. The complete
upstream license is preserved at `vendor/torch-webgpu/LICENSE` and staged into
`torch/webgpu/licenses` in the wheel.

## Emdawnwebgpu

`vendor/emdawnwebgpu` is the exact text-only package from Dawn release
`v20251002.162335`, Dawn revision
`01940842b667a7812d0e4ca0ef4367fbec294241`. Its archive SHA-512 is pinned in
`config/build.toml` and matches the Emscripten 5.0.3 remote-port pin.

The build uses its generated `webgpu.h` and Dawn-style `webgpu_cpp.h` headers.
The full Emdawn JavaScript library cannot be final-linked by a Pyodide side
module, so the project supplies a documented compute-only compatibility
profile through side-module `EM_JS`. Emdawnwebgpu includes BSD-3-Clause and
Emscripten dual MIT/UIUC-licensed portions. Its license texts and copyright
headers are preserved under `vendor/emdawnwebgpu`.
The relevant license texts are also staged into `torch/webgpu/licenses`.
