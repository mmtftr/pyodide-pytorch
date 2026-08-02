# Weight-only Q8 decode prototype

`webgpu::q8_linear` is an opt-in, decode-only linear operator. It reduces the
dominant projection-weight traffic without pretending that the existing
float32 `aten::linear` tensor is quantized. The experimental source is kept
as an explicit custom operator and has been added to the staged wheel build;
ordinary float32 `aten::linear` semantics remain unchanged.

## Format version 1

For a float32 row-major weight with shape `[N, K]`:

- `K` is positive and divisible by 128.
- Each `[row, group-of-128]` scale is the float32 maximum absolute weight.
- Each weight is `q = clamp(round(weight / scale * 127), -127, 127)`. A
  zero-absmax group uses all-zero `q` values and a zero scale.
- Four signed bytes are stored in one `torch.int32` word. Feature `4p + j`
  occupies bits `[8j, 8j + 7]`, so the packed shape is `[N, K/4]`.
- Scales are contiguous float32 with shape `[N, K/128]`.

The storage cost is 1 byte of weight plus 4/128 bytes of scale per original
weight: 1.03125 bytes, or 3.8788x smaller than float32. The shader uses
`unpack4x8snorm` and multiplies by the absmax, avoiding scalar sign-extension
code. Activations, bias, accumulation, scales, and output remain float32.

One fixed-size-32 subgroup computes four output rows. In each 128-value group,
one lane reads four adjacent activations and one packed word from each of four
rows. The input vector is therefore reused across four output dot products.

## Deliberate support boundary

The operator rejects, rather than silently rerouting, any call that violates
one of these conditions:

- WebGPU device with the `subgroups` feature and
  `subgroupMinSize == subgroupMaxSize == 32`;
- input, scales, optional bias, and output are float32; packed weight is int32;
- all operands are contiguous, on the same WebGPU device, and fit their bound
  GPUBuffer storage;
- exactly one flattened input row (`batch=1`, `sequence=1` for standard causal
  LM decode), positive `N` and `K`, and `K % 128 == 0`;
- packed weight `[N, K/4]`, scales `[N, K/128]`, optional bias `[N]`;
- group size 128, format version 1, and at most 65,535 workgroups.

This is not a prefill implementation. `WebGPUQ8Linear.forward` raises on more
than one flattened row. Conversion must therefore be restricted to a
single-token decode deployment, or wait until a packed-weight M>1 kernel is
available. There is no hidden CPU fallback and no retained float32 weight.

## Hardware evidence

The raw browser harness compares
`linear_gemv_q8_s4.wgsl` with the repository's current float32
`linear_gemv_subgroup_s4.wgsl`. It ran on 2026-08-01 in headless Chrome 150 on
an Apple M4 Pro (`metal-3`, fixed subgroup size 32), with zero WebGPU validation
errors. Nine alternating steady-state samples were taken after warm-up. Eleven
single-dispatch timestamp samples were also taken after a 32 MiB unrelated GPU
copy; that second measurement includes compute-pass boundary cost and is a
useful lower bound when dispatch overhead is not amortized.

| Projection `[K, N]` | fp32 steady ms | Q8 steady ms | steady speedup | fp32 scrubbed pass ms | Q8 scrubbed pass ms | pass speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B Q `[896, 896]` | 0.0225 | 0.0125 | 1.80x | 0.3133 | 0.2791 | 1.12x |
| Qwen2.5-0.5B gate/up `[896, 4864]` | 0.1000 | 0.0333 | 3.00x | 0.3635 | 0.2957 | 1.23x |
| Qwen2.5-0.5B down `[4864, 896]` | 0.1967 | 0.0533 | 3.69x | 0.4558 | 0.3251 | 1.40x |
| Qwen2.5-1.5B gate/up `[1536, 8960]` | 0.3000 | 0.0917 | 3.27x | 0.5431 | 0.3436 | 1.58x |

Exact GPU output versus a CPU reconstruction of the packed weights differed by
at most `1.2e-7` in these projection cases. Q8 output versus the original
synthetic float32 weights had normalized RMSE from 0.38% to 0.41%. That checks
the format and kernel, not language-model quality: real checkpoints still need
perplexity and generation evaluation before Q8 is a recommended default.

Reproduce the extended hardware run with:

```sh
WEBGPU_ADAPTER=hardware \
PACKED_GEMV_EXTENDED=1 \
CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
fnm exec --using=24.11.1 node tests/webgpu-packed-gemv-wgsl.mjs
```

SwiftShader reports subgroup size 4 in the same harness and is skipped after
the fail-closed host checks; the Q8 shader is never dispatched there.

## Wheel integration status

The prototype has all operator-specific pieces and is included in staging,
the PyTorch WebGPU source list, and the raw-shader release gate:

- `webgpu/llm_kernels/linear_gemv_q8_s4.wgsl`: validated shader;
- `webgpu/llm_kernels/q8_linear.cpp`: strict custom-op implementation and
  `webgpu::q8_linear` schema;
- `site/transformers_q8.py`: CPU packer, weight-owning `WebGPUQ8Linear`, and an
  all-or-nothing recursive conversion helper;
- `tests/webgpu-packed-gemv-wgsl.mjs`: hardware correctness, format, guard, and
  benchmark coverage;
- `tests/test_q8_linear_contract.py`: source/ABI/opt-in contract coverage.

The remaining acceptance sequence is:

1. Build once, then exercise `torch.ops.webgpu.q8_linear` in hardware Chrome
   with offset, bias, N-tail, dtype, shape, feature, and M>1 rejection cases.
2. Run a real Qwen checkpoint with a one-token input and compare logits and
   greedy output against the unpacked float32 model. Add multi-token prefill
   only after a packed M>1 kernel exists; do not keep fp32 weights merely to
   disguise the missing path.
3. Keep conversion explicit. Load an eval-mode CPU float32 checkpoint, call
   `convert_linear_modules_q8_`, release the original module references, and
   then enter the WebGPU decode loop. The converted state dict intentionally
   contains packed weights/scales instead of the original parameters.

Future work should compare group sizes on actual checkpoints, add a portable or
tiled M>1 packed kernel for prefill, and fuse Q8 gate/up projection with SwiGLU.
That last fusion should remove another dispatch while retaining the same packed
format.
