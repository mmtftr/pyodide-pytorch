# Third-party notices

## torch-webgpu

The experimental browser WebGPU backend in
`patches/pytorch/0010-add-experimental-browser-webgpu-backend.patch` adapts
the PrivateUse1 allocator and dispatch structure and binary TensorIterator
WGSL from
[jmaczan/torch-webgpu](https://github.com/jmaczan/torch-webgpu) commit
`a4369ff0f61f4e58cbffb048cee85047b33dacba`.

Those portions are licensed under the Apache License, Version 2.0. They have
been modified to replace the native Dawn transport with an Emscripten
JavaScript bridge, add browser-safe asynchronous readback, and integrate the
backend directly into the Pyodide PyTorch wheel. The full Apache-2.0 license
is included by the patch at `third_party/torch-webgpu/LICENSE` and is packaged
with the wheel's license files.
