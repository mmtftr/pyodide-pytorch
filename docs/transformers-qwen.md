# Hermetic Transformers WebGPU acceptance

The first library-level compatibility gate is a deterministic, model-only
forward pass, not an official checkpoint and not `model.generate()`. The
primary matrix is pinned to Transformers 4.46.3 and Hugging Face Hub 0.26.2
until every required architecture passes in a real browser:

| Family | Class | Required output |
| --- | --- | --- |
| Qwen2 | `Qwen2ForCausalLM` | logits |
| Llama | `LlamaForCausalLM` | logits |
| Mistral | `MistralForCausalLM` | logits |
| GPT-2 | `GPT2LMHeadModel` | logits |
| BERT | `BertModel` | last hidden state |
| Phi-3 | `Phi3ForCausalLM` | logits |
| OPT | `OPTForCausalLM` | logits |

The shared fixture uses a one-layer, width-16 model for each family, a fixed
seed, dropout disabled, SDPA attention, and checked-in pre-tokenized Long input
IDs. The weights are initialized deterministically by the pinned Transformers
and PyTorch versions. Each model runs on CPU first, then the same model and
inputs move to `webgpu`. Acceptance requires numerical parity, at least one GPU
dispatch, exactly one explicit output readback, zero CPU fallbacks, and zero
WebGPU validation errors. GPT-2 and BERT are required gates, not informational
probes.

## Browser-only dependency profile

`config/transformers-browser-requirements.txt` pins unchanged upstream,
pure-Python wheels with hashes:

- `filelock==3.32.0`
- `transformers==4.46.3`
- `huggingface-hub==0.26.2`

The remaining pure-Python/runtime dependencies come from the pinned Pyodide
314.0.2 package lock. The Rust `tokenizers` package is deliberately absent.
The test asserts both that its import spec is absent and that no distribution
metadata claims it is installed. Inputs are already tokenized, so no tokenizer
is needed for these forward passes.

Transformers 4.46.3 nevertheless imports its optional GGUF loader while
importing a model configuration. That loader imports `tokenizers`
unconditionally. The reusable
`site/transformers_browser_bootstrap.py` module disables only
`transformers.modeling_gguf_pytorch_utils` when `tokenizers` is absent and
makes `load_gguf_checkpoint` fail lazily with a clear browser-profile error.
It does not create a fake `tokenizers` package or alter wheel metadata. The
same checked-in module is installed by the playground worker and served to the
acceptance harness. The deployed worker fetches it from the Pages-root URL
`./transformers_browser_bootstrap.py`; the local-only harness maps that same
source file to `/runtime/transformers_browser_bootstrap.py`. If a real
`tokenizers` package exists, the bootstrap does nothing.

The bootstrap also exposes a separate, explicit
`enable_webgpu_rms_norm_fusion()` opt-in. Transformers 4.46.3 implements its
canonical `Qwen2RMSNorm`, `LlamaRMSNorm`, `MistralRMSNorm`, and `Phi3RMSNorm`
forwards as
`pow` → `mean` → epsilon addition → `rsqrt` → two multiplies. The
adapter sends only float32 `webgpu` inputs and weights through
`torch.nn.functional.rms_norm`, reaching the backend's fused
`aten::rms_norm` kernel. Every CPU, other-device, or other-dtype call delegates
to the captured upstream method, so the ordinary CPU reference path is not
rewritten.

This adapter deliberately supports exactly Transformers 4.46.3 and those
four classes in their canonical modules. It resolves and validates the whole
target set before making a process-wide class change. A different version,
missing class, unexpected forward signature, or missing PyTorch functional
raises `TransformersBrowserProfileError` and reports that no classes were
changed. Calls are idempotent and return JSON-serializable diagnostics showing
each target as `patched` or `already_enabled`. Supporting a later Transformers
release requires reviewing its implementations; this is not an adapter for
custom RMSNorm modules or arbitrary future models.

The independent `enable_webgpu_swiglu_fusion()` opt-in is narrower still. It
adapts only the canonical `Qwen2MLP`, `LlamaMLP`, and `MistralMLP` classes from
Transformers 4.46.3. A contiguous float32 WebGPU activation with exactly one
flattened row (the decode case) calls
`torch.ops.webgpu.fused_swiglu(input, gate_weight, up_weight, gate_bias,
up_bias)`. That operator computes both projections and `SiLU(gate) * up` in
one dispatch; the ordinary `down_proj` module then runs unchanged. The backend
selects a four-output specialization on adapters exposing fixed-size-32
subgroups and otherwise uses a portable 64-output workgroup shader.

Eligibility is deliberately fail-closed. Full-sequence or batched inputs,
noncontiguous or non-float32 tensors, non-WebGPU parameters, incompatible
biases, custom gate/up projections, a non-stock or in-place activation, Llama
tensor-parallel pretraining, autograd inputs, and child/global module hooks all
execute the captured upstream forward. CPU behavior is therefore untouched,
including projection and activation hooks. Enabling the adapter first validates
the exact package version, all three classes and signatures, `torch.nn.Linear`,
`torch.nn.SiLU`, and the custom op; it changes no class if any check fails and
is idempotent when repeated.

The focused browser probe preserves the stock CPU path as its baseline: three
linear dispatches plus SiLU and multiplication, or five dispatch-producing
operations. A one-row WebGPU call records one `webgpu::fused_swiglu` dispatch
and the unchanged down projection, exactly two dispatches total and three
saved. It requires numerical parity, one explicit readback, zero CPU
fallbacks, and a clean WebGPU validation scope for every adapted family. A
separate raw-Chrome gate compiles the checked-in WGSL and covers reduction
tails, nonzero storage offsets, absent/independent/one-sided biases, and output
sentinels. On an Apple M4 Pro it executed both the portable and subgroup-S4
variants in exactly one dispatch per case; SwiftShader exercised the portable
fallback because its subgroup size is four.

The independent `enable_webgpu_preallocated_kv_cache()` opt-in registers
`cache_implementation="webgpu_preallocated"` for pinned Transformers 4.46.3.
It supports the canonical sequential Qwen2, Llama, and Mistral cache contract
for greedy or sampling decode. Two fixed float32 allocations per layer hold K
and V for the requested capacity. Each update passes the standard WebGPU
int32/restricted-Long `cache_position` vector to one fused indexed-write
dispatch for both tensors. Repeated decode steps allocate no new GPU storage
and never concatenate the old prefix.

The cache class subclasses `DynamicCache`, rather than `StaticCache`, and
returns metadata-only `[..., :valid_length, :]` views after each update. The
pinned attention classes therefore retain their dynamic-cache mask behavior,
while fused SDPA receives the actual prefix length and does not score or read
uninitialized capacity. At decode prefix length `L`, two ordinary
DynamicCache concatenations use four copy dispatches and move
`16 * L * batch * kv_heads * head_dim` bytes; the paired indexed update uses
one dispatch and moves `16 * batch * kv_heads * head_dim` bytes. This is an
`L`-fold reduction in cache-update traffic and saves three dispatches per
layer per token.

Registration validates the exact package version, canonical DynamicCache
signatures, generation registries, and custom operator before changing
anything. It neither replaces upstream DynamicCache/StaticCache nor changes a
CPU path. The explicit implementation rejects non-WebGPU construction,
non-float32 or mismatched state shapes, missing/noncontiguous position vectors,
capacity overflow, and unsupported beam/batch transforms. Logical crop/reset
reuse storage. The sequential cache-position values used by stock generation
remain a caller invariant so the hot path needs no position readback.

The bootstrap's separate
`enable_webgpu_rotary_scaling_compatibility()` opt-in applies the same
fail-closed contract to the canonical `Qwen2RotaryEmbedding` and
`LlamaRotaryEmbedding` classes. In Transformers 4.46.3 their two expressions
`cos * attention_scaling` and `sin * attention_scaling` lift the Python number
to a CPU zero-dimensional tensor and select `aten::mul.Tensor`. That is both a
mixed-device error on WebGPU and unnecessary work for the default identity
scale. The adapter retains the captured upstream forward on CPU and every
other device. On WebGPU it reproduces the reviewed 4.46.3 forward, elides the
two operations when the finite Python scale is exactly `1.0`, and explicitly
selects `aten::mul.Scalar` for a non-identity scale. A non-numeric or non-finite
replacement fails closed. It validates both exact classes before changing
either one and reports idempotent, JSON-serializable diagnostics.

OPT has one additional pinned-library compatibility boundary. Its SDPA path
creates an all-one mask when the caller omits one, then evaluates
`torch.all(mask == 1)` in Python even though the answer is already known. The
exact-version `enable_webgpu_opt_sdpa_mask_compatibility()` adapter skips only
that GPU truth readback for an uncached full forward or one-token cached
forward with no caller mask. It returns the same generated attention mask and
`None` causal mask. CPU, supplied-mask, non-SDPA, attention-output, head-mask,
and multi-token cached calls execute the captured upstream method. The adapter
validates the canonical `OPTDecoder` method and signature, is idempotent, and
fails closed on any version or implementation change.

The Node harness serves the wheel, all dependency files, fixture, and bootstrap
from one loopback origin. Browser request interception rejects every other
origin, and the usual Hugging Face offline environment variables are set before
the library is imported. A passing test therefore cannot depend on the Hub or
another runtime network request.

The Build workflow downloads those three wheels with `--no-deps` and
`--require-hashes`, emits a machine-readable companion manifest, and makes the
seven-family browser matrix a required release gate. Pages repeats that verified
download from the same requirements file and deploys the manifest, wheels, and
fixture under `runtime/transformers/`. The playground verifies wheel sizes and
SHA-256 digests before loading them; neither its startup nor its tiny Qwen2
construction example contacts PyPI, pythonhost, or the Hugging Face Hub.

## Artifact baselines and diagnostics

The pre-integration `local-r5-opt` wheel establishes a useful failure baseline:
all five pinned models complete their deterministic CPU reference forward, and
the model-only bootstrap reports active with `tokenizers` unavailable. All five
then stop at their first Long transfer with:

```text
RuntimeError: WebGPU supports only torch.float32 and torch.int32
```

For Qwen2, Llama, and Mistral the first transfer is the Long model input. For
GPT-2 and BERT it occurs while moving registered Long buffers in the model.
This wheel predates the restricted-Long creation, embedding, copy, and cast
work, so the result is a baseline rather than a verdict on the integrated
implementation.

The acceptance result retains, per family, the complete CPU dispatcher trace,
the exact exception and traceback, output shape, dispatch/readback/fallback
deltas, and validation errors. Later artifact runs use those results to report
the first actual missing operator rather than infer one from source alone.

The same gate now measures the RMSNorm integration independently for all four
pinned classes. It requires the post-install CPU call to retain the six-op
upstream trace with no `aten::rms_norm`, while the stock class call on WebGPU
must contain one `aten::rms_norm` and exactly one dispatch. That replaces six
logical dispatches with one, a reduction of five per normalization. Each
one-layer tiny Qwen2/Llama/Mistral/Phi-3 fixture has three normalization calls, so
its model trace must contain three fused calls and records 15 avoided logical
dispatches per forward.

The current unfused browser path cannot be timed end to end: the pinned
upstream expression presents its Python epsilon as a CPU scalar tensor to
`aten::add.Tensor`, and mixed CPU/WebGPU arithmetic fails explicitly after the
first two component dispatches. The gate therefore uses the exact CPU
dispatcher trace as the composition baseline and measures the fused WebGPU
dispatch directly; it does not report a fabricated unfused WebGPU runtime.
These are dispatch-count assertions, not a hardware speed claim. SwiftShader
timing remains diagnostic, and public-size models have additional memory and
operator constraints.

A no-rebuild browser run against the available `local-r5-hf` wheel verifies
this adapter boundary even though that artifact does not yet pass the complete
current matrix. The original three focused classes retained the exact six-operation
CPU trace, produced numerically matching fused WebGPU output in one dispatch,
and completed their dedicated validation scope without an error. The tiny
Mistral forward also produced matching logits with 38 dispatches, one readback,
zero CPU fallbacks, and three `aten::rms_norm` calls (15 avoided logical
dispatches).

Before the rotary adapter, the same artifact stopped Qwen2 and Llama at
`cos * attention_scaling` with `WebGPU binary operation expects a WebGPU
tensor`. A no-rebuild Chrome hardware rerun with the adapter made both full
tiny forwards pass numerical comparison in 38 dispatches, exactly one
readback, zero CPU fallbacks, and zero validation errors. Their unmodified CPU
traces contain 13 `aten::mul.Tensor` calls, two more than Mistral's 11; all
three WebGPU traces contain five because the two identity rotary scales are
gone. Thus the adapter saves exactly two logical dispatches and their output
allocations per Qwen2/Llama rotary call. This existing-wheel result covers the
default `1.0` scale; non-identity profiles additionally require the integrated
`aten::mul.Scalar` kernel.

GPT-2 still cannot transfer one of its registered buffers on that older
artifact, and BERT still reaches unsupported Bool plumbing and a validation
error. Those remain full-matrix failures rather than being hidden by the
Qwen/Llama result, so the no-rebuild run is not a five-family acceptance pass.

For local diagnosis, setting `TRANSFORMERS_RMS_NORM_ONLY=1` on
`tests/transformers-webgpu.mjs` runs the same hermetic dependency and focused
RMSNorm plus rotary-scaling probes, including version, idempotency, preserved
CPU semantics, numerical parity, dispatch counts, fallbacks, and WebGPU
validation, then skips the unrelated model matrix. The legacy environment
variable name is retained for scripts that already use it. It is unset in CI,
so focused mode cannot weaken the required seven-family gate. On Chrome hardware
with `local-r5-hf`, each Qwen2/Llama rotary probe executed the remaining path
in eight dispatches and one readback, with no fallback or validation error,
while removing both upstream identity-scale dispatches.

## Latest-release compatibility probe

The primary forward matrix remains 4.46.3. Separately, on 2026-08-01, an
import-only probe checked the then-current [Transformers 5.14.1 release on
PyPI](https://pypi.org/project/transformers/5.14.1/), published 2026-07-16.
In an isolated environment with its required pure-Python dependencies but no
`tokenizers`, this import still failed:

```text
Qwen2Config
  -> configuration_utils
  -> modeling_gguf_pytorch_utils
  -> integrations.ggml
  -> ModuleNotFoundError: No module named 'tokenizers'
```

Installing the same model-only bootstrap before importing Transformers made
`Qwen2Config` import successfully with `tokenizers` still absent. Thus 5.14.1
does not remove the need for the workaround. This was a configuration-import
probe only; it is not a 5.14.1 model-forward compatibility claim. A 5.14.1
forward probe belongs after the integrated 4.46.3 matrix passes and must not
replace that primary gate.

## Scope after the forward gate

Stock `generate()` remains a later milestone. The preallocated cache removes
the quadratic DynamicCache append, but 4.46.3 still adds cache-position
creation, cumulative sums, comparisons, masked mutation, token selection, Bool
termination tensors, reductions, and local scalar checks beyond a single
forward call. Float32 last-dimension `argmax` token selection is now
source-complete; its focused browser case still awaits an integrated wheel. A
deliberately fixed-length loop with GPU argmax and no tensor-dependent host
termination is the smallest useful follow-up, followed by dynamic-cache
validation and only then the full stock generation loop.

### Exact stock-generation operator audit

A CPU dispatcher trace of the pinned one-layer Qwen2 fixture generated two
greedy tokens with Transformers 4.46.3 and PyTorch 2.13.0. It covered inferred
and explicit all-one attention masks with `use_cache=False`, plus inferred mask
with `use_cache=True`. The next work is ranked by execution order and by
whether kernels alone can make progress:

| Priority | Stock path | Exact ATen surface | Assessment |
| --- | --- | --- | --- |
| 0 | Python tensor-dependent branches | `_local_scalar_dense` after special-token `any`, SDPA's all-one-mask `all`, cached `cache_position[-1] >= ...`, uncached tensor-valued `arange` bounds, and `unfinished_sequences.max() == 0` | This is the architectural gate. The browser bootstrap now installs JSPI-safe tensor truth conversion, in addition to the JSPI `.cpu()`/`.item()` wrappers, but the dispatcher `_local_scalar_dense` kernel itself remains intentionally unsupported. Outside a JSPI-capable entrypoint synchronous inspection is impossible; stock generation must therefore enter through `runPythonAsync`/`callPromising`, while a browser-specific fixed-length loop can remove these branches entirely. |
| 1 | Special-token checks, inferred padding mask, and SDPA mask elision | Bool allocation/fill plus `isin.Tensor_Tensor`, `any`, `all`, `lt.Scalar`, `eq.Scalar`, `ne.Tensor`, `bitwise_not`, and Bool-to-Long `_to_copy` | This is the first kernel cluster reached by an unmodified default call. Passing an explicit all-one mask removes two `isin` calls but not special-token validation, position construction, or the model's `eq` + `all` mask check. |
| 2 | Initial cache positions and per-step position IDs | restricted-Long `cumsum`, Long `sub.Tensor`, `eq.Scalar`, and Long `masked_fill_.Scalar` with a Bool mask | `ones_like`, `new_ones`, restricted-Long creation, and Long `cat` are already source-complete. `prepare_inputs_for_generation` still executes comparison and masked fill even when every mask value is one. |
| 3 | EOS and max-length stopping state | Bool `full`, `bitwise_or.Tensor`, `bitwise_not`, `bitwise_and.Tensor`; Long `max`, `eq.Scalar`, `mul.Tensor`, `rsub.Scalar`, and `add.Tensor` | These run once per token in the stock loop. Removing EOS avoids the padding arithmetic, but MaxLength still constructs and combines Bool tensors and then performs the scalar termination check. |
| 4 | One-token cache input slicing | Long `ge.Scalar`, `index.Tensor`, and the scalar branch in `prepare_inputs_for_generation` | The opt-in preallocated cache replaces float32 DynamicCache concatenation with one paired indexed write and a valid-prefix view. On the second cached step stock code still compares and advanced-indexes with `cache_position`; these generation-control operators remain separate work. |

For the traced two-token no-cache call with an inferred mask, the generation
and SDPA plumbing alone performed three Long `cumsum` calls, two
`masked_fill_.Scalar` calls, five `isin.Tensor_Tensor` calls, two Bool `all`
reductions, and ten synchronous scalar extractions. Consequently, implementing
Bool and cumsum kernels without first choosing the Priority-0 control strategy
would not make stock `generate()` executable.

An official Qwen2.5 0.5B float32 checkpoint is also not a sensible baseline for
ordinary browsers: weights alone are about 2 GB and CPU load plus GPU upload
raises the peak further. Float16 or quantized storage, chunked upload, and
buffer-limit-aware parameter layouts are prerequisites for that experiment.
