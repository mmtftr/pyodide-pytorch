# Transformers architecture coverage audit

This audit expands the release gate from five to seven families without
pretending that every Hub repository is interchangeable. It uses the exact
browser dependency wheel
`transformers-4.46.3-py3-none-any.whl` (SHA-256
`a12ef6f52841fd190a3e5602145b542d03507222f2c64ebb7ee92e8788093aef`)
and the exact local Pyodide torch wheel used by the pre-integration browser
tests. The audited upstream implementations are:

- `Gemma2ForCausalLM`
- `Phi3ForCausalLM`
- `OPTForCausalLM`
- `BloomForCausalLM`
- `T5ForConditionalGeneration`

Phi-3 and OPT now live in the fixture's required `models` array alongside
Qwen2/Llama/Mistral/GPT-2/BERT. Gemma2, BLOOM, and T5 remain under
`architecture_audit_models`: discovering one of their known mask or position
gaps cannot weaken any of the seven mandatory rows. The audit page combines
both arrays to retain the same five-architecture CPU trace snapshot.

## Reproducible CPU dispatcher trace

`tests/transformers-architecture-audit.html` loads the same pinned, hash-bound
browser wheels with no tokenizer or network access. For every architecture it
constructs a deterministic tiny model and runs two CPU forwards. The trace is
accepted only when both the complete ATen count map and output tensor are
identical, the total/distinct counts match the pinned fixture snapshot, and
the architecture-specific critical operators remain present. Run it with the
existing hermetic browser server:

```sh
fnm exec --using=24.11.1 node tests/transformers-webgpu.mjs \
  TORCH_WHEEL FILELOCK_WHEEL TRANSFORMERS_WHEEL HUB_WHEEL \
  node_modules/pyodide tests/transformers-architecture-audit.html
```

The 2026-08-01 Pyodide CPU run produced:

| Family | Trace profile | Output | ATen calls | Distinct ops | Architectural signal |
| --- | --- | ---: | ---: | ---: | --- |
| Gemma2 | two layers, alternating sliding/global attention, SDPA | `[1, 4, 64]` | 251 | 37 | nine eager RMSNorm compositions, two SDPA calls, and explicit causal/sliding-mask construction |
| Phi-3 | one layer, unpadded explicit positions, SDPA | `[1, 4, 64]` | 99 | 27 | three eager RMSNorm compositions; otherwise the same central RoPE/SwiGLU/SDPA surface as the existing Llama-family rows |
| OPT | one layer, explicit positions and all-one mask, SDPA | `[1, 4, 64]` | 56 | 17 | three affine LayerNorm calls and one Bool `eq`/`all`/host-truth mask-elision branch |
| BLOOM | one layer, all-one mask, eager ALiBi attention | `[1, 4, 64]` | 88 | 34 | Long `cumsum`, tensor-power ALiBi construction, `baddbmm`, causal masking, and four LayerNorm calls |
| T5 | one encoder and one decoder layer with cross-attention | `[1, 3, 64]` | 285 | 46 | seven eager RMSNorm compositions, integer relative-position bucketing, and three manual softmax attention paths |

The CPU-specific
`aten::_scaled_dot_product_flash_attention_for_cpu` trace entries are not
WebGPU blockers: stock Transformers calls public SDPA, and PrivateUse1 reaches
the project `aten::scaled_dot_product_attention` implementation. Likewise CPU
`addmm` is normally intercepted by the project `linear` registration. The
dispatcher kernel-status field in the diagnostic is supporting information,
not an acceptance verdict: composite registrations may report a computed
PrivateUse1 kernel even when a nested operation still lacks a browser-safe
implementation.

## Family-by-family boundary

### Phi-3: promoted decoder family

The unpadded, full-sequence Phi-3 path is structurally closest to Qwen2 and
Llama. Its trace contains `linear`, RoPE (`cos`, `sin`, views, `cat`, elementwise
arithmetic), SiLU gating, and SDPA, all of which are already in the integrated
source surface. The exact-version browser adapter now includes canonical
`Phi3RMSNorm`; float32 WebGPU forwards replace its three six-op normalization
compositions with three fused `rms_norm` calls while CPU continues through the
captured upstream method. This saves 15 logical dispatches per tiny forward.
The unpadded, explicit-position configuration is consequently a mandatory
numerical-parity row. Padded attention and sliding-window configuration still
require the mask tranche below.

### OPT: promoted classic LayerNorm decoder

With explicit position IDs and an all-one attention mask, OPT uses the already
implemented affine LayerNorm, ReLU, linear, embedding, and SDPA paths. Its one
tensor-dependent branch is Transformers' all-one-mask elision check:
`eq.Scalar` -> Bool `all` -> Python truth. A version-checked WebGPU-only
adapter recognizes only the two shape cases for which the answer is known
without inspecting tensor data: an uncached full forward or a one-token cached
forward with no caller mask. It returns the same generated all-one mask and
`None` causal mask that upstream computes, avoiding the host truth readback.
CPU, caller-provided masks, non-SDPA attention, attention outputs, head masks,
and multi-token cached calls retain the captured upstream method. That narrow
profile makes the explicit-position, unpadded OPT forward a mandatory row;
implicit positions add Long `cumsum`, and padded masks add the general
mask-construction surface.

### Gemma2: mask construction plus a distinct RMSNorm contract

Gemma2 always builds a four-dimensional causal mask and alternates
sliding-window and global layers. The exact trace includes `full`, `triu`,
`arange`, `gt.Tensor`, `mul_.Tensor`, `ones_like` Bool, `tril`, and
`where.self`. Creation and arange are present, but triangular fill, general
comparison, selection, and tensor in-place arithmetic are not in the verified
browser subset.

Gemma2 normalization also cannot be added blindly to the existing adapter.
Its learned parameter is an offset and upstream computes
`normalized * (1 + weight)`, whereas `torch.nn.functional.rms_norm` treats its
weight as the complete multiplier. A correct fusion needs either an
offset-weight mode in the backend or a reviewed adapter that preserves that
contract without allocating `1 + weight` on every token. The two-layer audit
has nine normalization calls, so a correct fusion saves 45 logical dispatches.

### BLOOM: ALiBi and eager attention are the main gap

BLOOM does not select an SDPA attention class in this pinned release. Its ALiBi
path performs restricted-Long `cumsum` and arithmetic, Float-by-Int
`pow.Tensor_Tensor`, then eager attention through `baddbmm`, softmax, and BMM.
The causal mask additionally uses triangular construction, comparison, and
`masked_fill.Scalar`. LayerNorm and the final softmax/BMM primitives are
already available; ALiBi construction and `baddbmm` are not.

A correctness-first route is generic Long position arithmetic plus an
alias-safe `baddbmm` implementation built on the existing BMM/binary kernels.
A faster later route is a reviewed BLOOM attention adapter or fused kernel that
feeds combined ALiBi and causal bias into SDPA, avoiding the materialized score
round trip. The custom tanh GELU expression is supported compositionally but is
also a worthwhile fusion target after the forward path works.

### T5: relative-position bucketing is the broadest remaining surface

T5 exercises encoder self-attention, decoder self-attention, and
cross-attention. Its relative-position bucket code needs Long comparisons,
absolute value and minimum, Float logarithm, Float-to-Long truncating cast,
Long `where`, and in-place Long addition. Decoder masking adds `triu`, tensor
comparison, in-place multiplication, and `masked_fill.Scalar`. Its seven
`T5LayerNorm` calls use ordinary multiplicative RMSNorm semantics and could
save 35 logical dispatches through an exact-version fused adapter, but that
optimization alone does not make relative attention executable.

Once the position-bias path works, T5's three manual score/softmax/BMM paths
are a strong performance target. The existing SDPA kernel already accepts an
additive float mask, so a reviewed T5 adapter can combine position bias and
mask and use `scale=1.0`; cross-attention and cache behavior must remain
separate acceptance cases.

## Ranked next implementation tranches

1. **Packed-Bool mask construction.** Implement broadcast `gt.Tensor`,
   `gt.Scalar`, and `lt.Scalar`; float/Long `where.self`; functional and
   in-place `masked_fill.Scalar`; Float/Bool `triu` and `tril`; and alias-safe
   `mul_.Tensor`/`add_.Tensor`. This is the widest correctness tranche: it
   unlocks Gemma2's causal/sliding masks and is shared by padded Phi-3/OPT,
   BLOOM, T5, and stock generation. Bool outputs must retain one-byte ATen
   storage with race-free packed WGSL writes.

2. **Restricted-Long position arithmetic.** Add Long `cumsum`, same-dtype
   add/sub/mul (including scalar reverse-subtract), Long `abs`/`minimum`, and a
   checked Float-to-restricted-Long conversion. This removes implicit-position
   workarounds, supplies BLOOM ALiBi, and covers most of T5 relative bucketing.
   Every producer must preserve the signed-int32-value invariant in ordinary
   eight-byte Long storage.

3. **Normalization adapters/fusions.** Phi-3 is now covered by the reusable
   exact-version RMSNorm profile. Add a separately reviewed T5 target, then
   implement Gemma2's offset-weight semantics explicitly. The audit measures
   35 and 45 additional avoidable logical dispatches respectively.

4. **Architecture-specific attention.** Implement or decompose `baddbmm` for
   BLOOM, then route BLOOM ALiBi and T5 relative bias through fused SDPA where
   numerical parity permits. SDPA is already fused for Gemma2, Phi-3, and OPT,
   so another generic causal SDPA kernel is not the immediate breadth win.

5. **Decode-only MLP fusion after the shared forward rows pass.** The project
   `webgpu::fused_swiglu(input, gate_weight, up_weight, gate_bias=None,
   up_bias=None)` operator performs both GEMVs plus `silu(gate) * up` in one
   dispatch for exactly one flattened float32 decode row. Phi-3 uses the same
   math but stores gate/up as two row ranges of one `gate_up_proj` weight, so a
   version-checked adapter can pass those contiguous views to the fused op and
   retain upstream code for prefill. Gemma2 uses gated GELU and BLOOM uses a
   hand-written tanh GELU, so neither may use the SwiGLU kernel unchanged.
   These fusions reduce bandwidth and eager dispatch overhead but should not
   replace the mask/position kernels required for model correctness.

## Practical support claim

The useful target is: pinned, pure-Python Transformers model classes whose
float32 eval forward lowers to the accepted operator profile, using
pre-tokenized signed-int32-valued Long IDs, full-sequence or functional dynamic
cache execution, and no CPU fallbacks inside a JSPI-capable browser entrypoint.

It is not a claim that arbitrary Hub repositories work. Remote custom code,
tokenizers, checkpoint download/load peaks, float16/bfloat16, quantization,
MoE routing, sampling, static/sliding cache mutation, training/autograd, and
arbitrary tensor-dependent Python control remain separate boundaries.
Standard vision towers are not nearly free: ViT/CLIP-style patch embeddings
require convolution, which is outside the compiled browser operator set, so a
vision fixture would currently measure a known `conv2d` wall rather than add
useful coverage.
