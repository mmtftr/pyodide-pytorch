# Browser WebGPU operator support

The status describes the pinned browser build, not native torch-webgpu. All
model arithmetic is float32 and inference-only; token indices may be int32.
Position and token plumbing may use ordinary eight-byte Long tensors when
every value fits signed int32; those values use canonical low/high words with
a sign-extended high word.
“Browser test” means
the operation is exercised by `tests/webgpu.html` against stock Pyodide and a
real browser `GPUDevice`; it is not inferred from successful compilation.

| Status | Operators / area | Evidence or reason |
| --- | --- | --- |
| Verified browser WebGPU implementation | Allocation/destruction; float32/int32 CPU upload and async readback; GPU copy; `view`/slice/transpose metadata; noncontiguous `contiguous`/`clone`; `cat`; `add.Tensor`, `mul.Tensor`, `neg`, `relu`, `gelu`, `silu`; 2-D `mm`, 3-D `bmm`/`matmul`, fused tiled `linear` with optional bias and a SIMD32 subgroup decode path; one-row fused gate/up GEMV plus SwiGLU; int32 `embedding`; stable affine `native_layer_norm` with output/mean/rstd, plus `rms_norm`; single-dimension float32 `mean.dim`; same-device `_to_copy` with packed Bool to canonical restricted-Long and int32/limited-Long to float32; fused float-mask/causal `scaled_dot_product_attention`, including GQA; contiguous last-dimension `softmax.int` | The Chromium gate against stock Pyodide 314.0.2 compares isolated operations, stock Transformers RMSNorm/position-cast compositions, composed rotary encoding, and a complete tiny GPT block with CPU PyTorch. Exact-version browser adapters make pinned Qwen2/Llama/Mistral/Phi-3 stock classes use fused RMSNorm only for float32 WebGPU calls, route one-row Qwen2/Llama/Mistral MLP calls through fused SwiGLU, make Qwen2/Llama rotary scaling device-safe, and avoid OPT's known all-one-mask GPU truth readback. The decode MLP probe records two dispatches including `down_proj`, versus five dispatch-producing operations upstream (three saved). Raw Chrome tests execute both portable and fixed-32 subgroup SwiGLU shaders when supported and require one dispatch per case. Focused gates preserve the six-op CPU RMSNorm trace while measuring one fused dispatch (five saved), and preserve the two CPU `mul.Tensor` rotary scales while eliding both for the default identity profile. The latter completes in eight dispatches plus one readback with zero fallbacks or validation errors on Chrome hardware. The focused LayerNorm gate covers a noncontiguous rank-four input, a two-dimensional noncontiguous affine weight, a 518-value reduction tail, all three native outputs, empty/error behavior, and one-submission command batching. A raw Chrome test executes the staged WGSL against large-offset inputs and the no-affine repeated-read binding. The decoder uses exactly 15 dispatches and 19 tensor allocations per forward (34 total GPUBuffer allocations including one transient uniform buffer per dispatch). An explicit batch records those 15 logical command buffers into one physical browser encoder and command buffer. CPU fallbacks must remain zero. |
| Compiles; browser verification required | Restricted-Long allocation, checked CPU upload and async readback; two-word Long `clone`/`contiguous`/`cat`; Long `embedding`; byte-packed Bool allocation/upload/readback/fill/clone/copy; integer `eq.Scalar`/`ne.Tensor`; Bool `all`/`any`/`bitwise_not`/`mul.Tensor`; restricted-Long `lt.Scalar`/`cumsum`/`isin`; float32 `masked_fill.Scalar`; bounded `gt.Tensor`, `triu`/`tril`, `where.self`, and Float-by-Bool `mul_.Tensor`; float32/int32/restricted-Long `arange`; scalar `fill_` and its `full`/`ones` composites; float32 last-dimension `argmax` returning canonical Long; paired preallocated float32 K/V indexed writes; opt-in group-128 signed-Q8 one-row linear; float32 `add/sub/mul/div.Scalar`, generated scalar-out, and scalar-in-place overloads; `sub.Tensor`, `div.Tensor`, alias-safe `add/mul/sub/div.out`; `relu.out`, `gelu.out`, `silu.out`; `cos`, `sin`, `tanh`, `exp`, `abs`, `rsqrt`, `log` and `.out`; `pow.Tensor_Scalar`; `.out` variants for matrix operations; `softmax.int_out`, `log_softmax.int`; broader ranks and shapes accepted by `matmul`; JSPI-gated `to_cpu_sync`, WebGPU `.cpu()`/`.item()`, and scalar truth testing | These registrations and wrappers resolve to genuine GPU paths and have focused cases in the browser suite, but this exact entry point, shader variant, or shape has not completed the numerical browser gate for the current wheel. The paired cache-update WGSL is independently exact on Apple M4 Pro Metal 3 and SwiftShader for int32/canonical restricted-Long positions, noncontiguous state strides, offsets, tails, and invalid-index containment, with one dispatch and zero validation errors per case. The Q8 shader is independently exact against CPU dequantization on Apple M4 Pro and reduces stored projection weights by 3.88x; conversion is explicit and decode-only. Bool preserves ATen's one-byte storage and uses four packed bytes per `u32`; word-owned or serialized writes avoid byte races. Synchronous readback additionally requires `can_run_sync()` and entry through `runPythonAsync()` or `callPromising()`. |
| Unsupported: missing kernel | Sigmoid; reductions needed by arbitrary model code; boolean attention masks; sampling/top-k; fused model-specific kernels; compiler subsystem and `torch.compile` | No complete browser-safe WGSL path is present in the imported or project-owned kernel set. |
| Unsupported: CPU fallback | In-place/output `masked_fill` overloads; `gather`, `scatter`, indexing/index-select; unsupported `where` overloads; dimension-specific sum and multi-dimension/general mean; dimension max/min; MoE scatter/top-k/nonzero paths; and comparison, reduction, `isin`, or bitwise overloads outside the bounded entries above | Upstream implementations synchronously move data through CPU. They are intentionally not compiled or registered. |
| Unsupported: host inspection outside a JSPI entrypoint | `_local_scalar_dense`, native WebGPU-to-CPU `copy_`, `.cpu()`/`.item()`/scalar truth entered through `runPython()` or a direct synchronous PyProxy call, and runtimes where `can_run_sync()` is false | Browser WebGPU exposes mapping asynchronously. Use `await torch.webgpu.to_cpu_async(tensor)` universally, or the synchronous wrappers only from a JSPI-capable async entrypoint. |
| Unsupported: incomplete browser contract | Model/activation dtypes other than float32, Bool outside the listed mask-control primitives and Bool-to-restricted-Long generation cast, Long values outside signed-int32 range, dtype conversions other than Bool to restricted-Long, int32/restricted-Long to float32, or same-dtype copies, non-last-dimension or non-contiguous softmax, mixed CPU/WebGPU arithmetic, autograd, upstream StaticCache/SlidingWindowCache and beam cache transforms, general/upstream quantization APIs, training | These paths can trigger upstream conversions, copies, or registrations outside the verified compute subset and therefore fail explicitly. Ordinary DynamicCache retains the project `cat` path; the exact-version Qwen2/Llama/Mistral adapter adds an opt-in fixed-capacity DynamicCache-compatible path with one paired indexed write and valid-prefix views. The separate `webgpu::q8_linear` format is an explicit decode-only custom operator, not support for arbitrary quantized tensors or Transformers quantizers. |

This is enough for the tested full-sequence, batch-one GPT decoder profile and
the central Llama-family primitives. It is not a claim that arbitrary model
repositories run unchanged: tokenization, model loading, RoPE conventions,
cache management, sampling, unsupported dtypes, and custom operators remain
application-specific compatibility boundaries.

## Linear performance benchmark

`tests/webgpu-linear-benchmark.html` compares the portable M=1 GEMV with a
four-output SIMD32 subgroup reduction. The subgroup path is enabled only when
the device has the `subgroups` feature and reports a fixed subgroup size of 32;
all other devices retain the portable shader. Random numerical tests cover K/N
tails, storage offsets, bias, and weights through a 180,355,072-byte binding.

`tests/webgpu-decode-scaling.html` measures the four distinct projection
shapes in stock Qwen2.5 blocks and applies the seven-call formula: two H-to-H,
two H-to-GQA-KV, two H-to-intermediate, and one intermediate-to-H. On an Apple
M4 Pro with Chrome 150, the optimized hot-repeat projection-only estimate
measured 24.17, 14.59, 8.43, and 5.57 wall-clock tokens/s for the 0.5B, 1.5B,
3B, and 7B configurations. Against the frozen pre-subgroup wheel, 3B improved
from 6.38 to 8.43 tokens/s (1.32x wall-clock and 1.47x GPU timestamp
throughput); 0.5B improved only 1.02x wall-clock despite a 1.25x GPU gain
because fixed eager overhead dominates its smaller matrices.

These are isolated hot-repeat fp32 projection estimates, not end-to-end model
throughput. They exclude the vocabulary head, full-model weight residency,
attention, normalization and activation, KV-cache traffic, sampling, and
tokenization. SwiftShader timings remain diagnostic only and are not used as a
performance acceptance threshold.
