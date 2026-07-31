# Browser WebGPU operator support

The status describes the pinned browser build, not native torch-webgpu. All
implemented kernels are float32-only and inference-only. “Browser test” means
the operation is exercised by `tests/webgpu.html` against stock Pyodide and a
real browser `GPUDevice`; it is not inferred from successful compilation.

| Status | Operators / area | Evidence or reason |
| --- | --- | --- |
| Verified browser WebGPU implementation | Allocation/destruction, CPU upload, GPU copy, async readback, `view`/slice/transpose metadata, `add.Tensor`, `mul.Tensor`, `relu`, 2-D `mm`, contiguous last-dimension `softmax.int` | The built wheel completed the Chromium WebGPU gate against stock Pyodide 314.0.2. Numerical results matched CPU references, all nine kernel executions incremented dispatch/submission counters, and the CPU-fallback counter remained zero. |
| Compiles; browser verification required | `sub.Tensor`, `div.Tensor`, `add/mul/sub/div.out`; `relu.out`, `gelu`, `gelu.out`, `silu`, `silu.out`; `cos`, `sin`, `tanh`, `exp`, `abs`, `rsqrt`, `neg`, `log` and `.out`; `pow.Tensor_Scalar`; `mm.out`; `softmax.int_out`, `log_softmax.int` | These registrations resolve to genuine pinned WGSL paths and link in the wheel, but this exact entry point or shader variant has not yet completed the numerical browser gate. |
| Unsupported: missing kernel | Sigmoid; general `matmul`/batched matmul not reducible to the verified 2-D `mm`; causal/masked attention dependencies; compiler subsystem and `torch.compile` | No complete pinned WGSL path in the imported narrow source set. |
| Unsupported: CPU fallback | `masked_fill`, `gather`, `scatter`, `where`, `argmax`, indexing/index-select; dimension-specific mean/sum; cumsum; dimension max/min; MoE scatter/top-k/nonzero paths; comparison reductions and `isin`/bitwise paths | Upstream implementations synchronously move data through CPU. They are intentionally not compiled or registered. Browser tests require `argmax` to fail. |
| Unsupported: synchronous readback | `.cpu()`, `.item()`, `_local_scalar_dense`, native WebGPU-to-CPU `copy_`, and any path using `MapAsync` plus `Instance::WaitAny` | Single-threaded browser WebGPU exposes mapping asynchronously; use `await torch.webgpu.to_cpu_async(tensor)`. |
| Unsupported: incomplete browser contract | Non-float32 tensors and conversions, non-last-dimension or non-contiguous softmax, mixed CPU/WebGPU arithmetic, autograd | These paths can trigger upstream conversions, copies, or registrations outside the verified compute subset and therefore fail explicitly. |

Creation, reductions, comparisons, and embedding contain genuine upstream WGSL
implementations, but they are not yet compiled into this first browser slice.
They remain candidates after their dtype, zero-size, limit, and registration
behavior is audited in a browser.
