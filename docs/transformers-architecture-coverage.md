# Transformers WebGPU architecture coverage

The browser release gate covers ten pinned Transformers families without
claiming that every Hub repository or PyTorch operator is interchangeable. It
uses the hash-bound `transformers==4.46.3` model-only dependency set, the local
Pyodide torch wheel under test, pre-tokenized inputs, deterministic tiny
configurations, and no network access.

The supported architecture profile is complete only when all ten browser
forwards match CPU PyTorch, issue at least one WebGPU dispatch, report exactly
one final readback, produce no CPU fallbacks, and produce no WebGPU validation
errors.

| Family | Attention/profile | Gate | Distinct coverage signal |
| --- | --- | --- | --- |
| Qwen2 | SDPA decoder | Main matrix | RoPE, fused RMSNorm, fused one-row SwiGLU |
| Llama | SDPA decoder | Main matrix | RoPE, fused RMSNorm, fused one-row SwiGLU |
| Mistral | SDPA decoder | Main matrix | RoPE, fused RMSNorm, fused one-row SwiGLU |
| GPT-2 | eager decoder | Main matrix | affine LayerNorm and linear compatibility |
| BERT | SDPA encoder | Main matrix | Bool mask-elision compatibility |
| Phi-3 | SDPA decoder | Main matrix | explicit positions and fused RMSNorm |
| OPT | SDPA decoder | Main matrix | affine LayerNorm and mask-truth avoidance |
| BLOOM | eager ALiBi decoder | Main matrix | Long ALiBi arithmetic and fused `baddbmm` |
| T5 | eager encoder/decoder/cross-attention | Main matrix | Long relative-position bucketing and three manual attention paths |
| Gemma2 | two-layer alternating sliding/global SDPA | Dedicated gate | causal/sliding masks, offset RMSNorm, softcaps, and Long control flow |

`tests/transformers-webgpu.html` requires the nine main-matrix rows.
`tests/transformers-gemma2-webgpu.html` keeps Gemma2 separate because its
two-layer fixture and exact-version adapter intentionally exercise a larger
mask/softcap surface. The separation is organizational, not a weaker status.

## Reproducible dispatcher audit

`tests/transformers-architecture-audit.html` loads the same pinned wheels and
runs two deterministic CPU forwards for the five architectures added during
the coverage expansion. It requires identical outputs and complete ATen count
maps across both passes, exact total/distinct counts, and the critical
architecture-specific operators. Run it with the hermetic browser harness:

```sh
node tests/transformers-webgpu.mjs \
  TORCH_WHEEL FILELOCK_WHEEL TRANSFORMERS_WHEEL HUB_WHEEL \
  node_modules/pyodide tests/transformers-architecture-audit.html
```

The pinned trace snapshot is:

| Family | Output | ATen calls | Distinct ops | Critical surface |
| --- | ---: | ---: | ---: | --- |
| Gemma2 | `[1, 4, 64]` | 251 | 37 | `gt`, `tril`, `triu`, Long/Bool `where` mask construction |
| Phi-3 | `[1, 4, 64]` | 99 | 27 | eager RMSNorm composition, RoPE, SwiGLU, SDPA |
| OPT | `[1, 4, 64]` | 56 | 17 | affine LayerNorm and Bool `eq`/`all` branch |
| BLOOM | `[1, 4, 64]` | 88 | 34 | Long `cumsum`, mixed tensor power, `baddbmm`, causal masking |
| T5 | `[1, 3, 64]` | 285 | 46 | Long bucket arithmetic, checked cast, three manual softmax/BMM paths |

CPU-only dispatcher entries such as
`aten::_scaled_dot_product_flash_attention_for_cpu` are trace metadata, not
WebGPU blockers: the PrivateUse1 forward reaches the project SDPA registration.
Likewise, CPU `addmm` is intercepted by the browser `linear` compatibility
path. Browser numerical parity and zero fallbacks are the acceptance verdict.

## Kernels that close the expanded matrix

The shared mask tranche provides packed-Bool comparisons and selection,
`masked_fill`, triangular construction, and alias-safe in-place arithmetic.
Gemma2, BLOOM, T5, padded decoder masks, and generation all use that surface.

The restricted-Long tranche preserves ordinary eight-byte `torch.int64`
storage while accepting values in the signed-int32 profile. It includes
tensor/scalar add, subtract, multiply, reverse subtract, `abs`, `minimum`,
`lt`/`gt`, Long `where`, `cumsum`, and checked truncating float conversion.
Zero-dimensional CPU scalar tensors created by ATen operator promotion are
unboxed into shader uniforms; the WebGPU tensor is never transferred to CPU.
Noncanonical inputs are contained rather than silently reinterpreted. These
operators cover BLOOM ALiBi and T5 relative-position buckets.

Mixed float/integer multiplication and tensor power cover ALiBi scaling. A
fused float32 `baddbmm` handles BLOOM's eager score path, including broadcast
`self`, `alpha`, and `beta`. Existing softmax and BMM kernels complete BLOOM
and T5 eager attention.

## Exact scope of “complete”

Complete coverage means every lowering used by the ten deterministic browser
forwards above is implemented and enforced by the release gate. The supported
profile is float32 eval inference with pre-tokenized signed-int32-valued Long
IDs, full-sequence or explicitly supported cache execution, and JavaScript
entry through `runPythonAsync()` or PyProxy `callPromising()` so JSPI can keep
promise boundaries out of the Python model call graph.

It is not a claim that arbitrary Hub repositories work. Remote custom code,
tokenizers, checkpoint download/load peaks, float16/bfloat16, general
quantization, MoE routing, sampling/top-k, static or sliding cache mutation,
training/autograd, convolutions, and arbitrary tensor-dependent Python control
remain separate compatibility boundaries and fail closed when no browser-safe
kernel exists.
