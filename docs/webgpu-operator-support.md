# Browser WebGPU operator support

The status describes the pinned browser build, not native torch-webgpu. All
model arithmetic is float32 and inference-only; token indices may be int32.
“Browser test” means
the operation is exercised by `tests/webgpu.html` against stock Pyodide and a
real browser `GPUDevice`; it is not inferred from successful compilation.

| Status | Operators / area | Evidence or reason |
| --- | --- | --- |
| Verified browser WebGPU implementation | Allocation/destruction; float32/int32 CPU upload and async readback; GPU copy; `view`/slice/transpose metadata; noncontiguous `contiguous`/`clone`; `cat`; `add.Tensor`, `mul.Tensor`, `neg`, `relu`, `gelu`, `silu`; 2-D `mm`, 3-D `bmm`/`matmul`, `linear`; int32 `embedding`; `native_layer_norm`, `rms_norm`; fused float-mask/causal `scaled_dot_product_attention`, including GQA; contiguous last-dimension `softmax.int` | The Chromium gate against stock Pyodide 314.0.2 compares isolated operations, composed rotary encoding, and a complete tiny GPT block with CPU PyTorch. Dispatch/submission counters must increase and CPU fallbacks must remain zero. |
| Compiles; browser verification required | `sub.Tensor`, `div.Tensor`, `add/mul/sub/div.out`; `relu.out`, `gelu.out`, `silu.out`; `cos`, `sin`, `tanh`, `exp`, `abs`, `rsqrt`, `log` and `.out`; `pow.Tensor_Scalar`; `.out` variants for matrix operations; `softmax.int_out`, `log_softmax.int`; broader ranks and shapes accepted by `matmul` | These registrations resolve to genuine WGSL paths and link in the wheel, but this exact entry point, shader variant, or shape has not completed the numerical browser gate. |
| Unsupported: missing kernel | Sigmoid; reductions needed by arbitrary model code; boolean attention masks; sampling/top-k; fused model-specific kernels; compiler subsystem and `torch.compile` | No complete browser-safe WGSL path is present in the imported or project-owned kernel set. |
| Unsupported: CPU fallback | `masked_fill`, `gather`, `scatter`, `where`, `argmax`, indexing/index-select; dimension-specific mean/sum; cumsum; dimension max/min; MoE scatter/top-k/nonzero paths; comparison reductions and `isin`/bitwise paths | Upstream implementations synchronously move data through CPU. They are intentionally not compiled or registered. Browser tests require `argmax` to fail. |
| Unsupported: synchronous readback | `.cpu()`, `.item()`, `_local_scalar_dense`, native WebGPU-to-CPU `copy_`, and any path using `MapAsync` plus `Instance::WaitAny` | Single-threaded browser WebGPU exposes mapping asynchronously; use `await torch.webgpu.to_cpu_async(tensor)`. |
| Unsupported: incomplete browser contract | Model/activation dtypes other than float32, token indices other than int32, dtype conversions, non-last-dimension or non-contiguous softmax, mixed CPU/WebGPU arithmetic, autograd, KV-cache mutation, quantization, training | These paths can trigger upstream conversions, copies, or registrations outside the verified compute subset and therefore fail explicitly. |

This is enough for the tested full-sequence, batch-one GPT decoder profile and
the central Llama-family primitives. It is not a claim that arbitrary model
repositories run unchanged: tokenization, model loading, RoPE conventions,
cache management, sampling, unsupported dtypes, and custom operators remain
application-specific compatibility boundaries.
