# Browser LLM kernels

This directory contains project-owned WebGPU kernels needed by the first
decoder-only transformer inference profile. It is intentionally separate from
`vendor/`: none of these files are part of the pinned torch-webgpu or
Emdawnwebgpu snapshots.

The normal staging step copies this directory to
`third_party/pyodide-pytorch-webgpu/llm_kernels` in the patched PyTorch source
tree and generates `embedded_shaders.h` from the WGSL files. Shader source is
therefore compiled into the PyTorch side module; the wheel does not perform
runtime filesystem reads or network fetches.

The profile is inference-only. Float32 is used for model state and
activations, int32 for token indices, and ordinary eight-byte Long storage for
the signed-int32-valued position/generation subset. Restricted Long values are
stored as low/high 32-bit words with a canonical sign-extended high word;
CPU uploads and project-owned producers validate the range. Bool retains one
byte per ATen element and is exposed to WGSL as four packed bytes per `u32`.
The operators reject unsupported layouts and dtypes rather than reading
tensors back through the CPU.

## Implemented vertical slice

| Source | Registered behavior |
| --- | --- |
| `boolean.cpp`, `eq_scalar.wgsl`, `ne_tensor.wgsl`, `all_bool.wgsl` | Bounded numeric `eq.Scalar`/integer `ne.Tensor` with byte-packed Bool output and a strided Bool `all` reduction |
| `argmax.cpp`, `argmax.wgsl` | Float32 last-dimension `argmax` with strided reads and canonical restricted-Long output for greedy token selection |
| `creation.cpp`, `arange.wgsl`, `fill.wgsl` | Float32/int32/restricted-Long `arange` plus contiguous scalar fill, including packed Bool, used by `full`/`ones` composites |
| `copy.cpp`, `strided_copy.wgsl` | Raw packed-Bool, 32-bit, or two-word Long clone/contiguous materialization and concatenation for ranks up to eight |
| `embedding.cpp`, `embedding.wgsl` | Float32 embedding weights with contiguous int32 or canonical restricted-Long indices |
| `generation_control.cpp`, `any_bool.wgsl`, `bitwise_not_bool.wgsl`, `long_lt_scalar.wgsl`, `mul_bool_tensor.wgsl` | Bool `any`/`bitwise_not`/`mul.Tensor` and restricted-Long `lt.Scalar`/`gt.Scalar` generation control |
| `generation_long.cpp`, `long_cumsum.wgsl`, `long_isin.wgsl` | Restricted-Long last-dimension `cumsum` and tensor membership with packed-Bool output |
| `gemma_rms_norm.cpp`, `gemma_rms_norm.wgsl` | Fused float32 Gemma2 offset-weight RMSNorm, preserving `normalized * (1 + weight)` |
| `kv_cache.cpp`, `kv_cache_update.wgsl` | One-dispatch paired K/V writes into a fixed-capacity float32 decode cache using int32 or restricted-Long positions |
| `long_arithmetic.cpp`, `long_arithmetic.wgsl`, `mixed_pow.wgsl` | Broadcast restricted-Long tensor/scalar arithmetic, `abs`, `minimum`, checked profile containment, and mixed float/integer tensor power for T5 buckets and BLOOM ALiBi |
| `masking.cpp`, `gt_tensor.wgsl`, `triangular.wgsl`, `where_float.wgsl`, `mul_bool_inplace.wgsl` | Bounded broadcast comparison, triangular masks, float/Long selection, and in-place Float-by-Bool or restricted-Long mask construction |
| `masked_fill.cpp`, `masked_fill_scalar.wgsl` | Broadcast float32 `masked_fill.Scalar` with strided packed-Bool masks |
| `q8_linear.cpp`, `linear_gemv_q8_s4.wgsl` | Opt-in one-row group-128 signed-Q8 linear with fixed-32 subgroup reduction, int32-packed weights, and per-row-group float32 scales |
| `matmul.cpp`, `bmm.wgsl`, `baddbmm.wgsl` | Strided 3-D BMM, fused broadcast `baddbmm`, equal-rank batched matmul, and linear composition through the existing 2-D MM kernel |
| `normalization.cpp`, `layer_norm.wgsl`, `rms_norm.wgsl` | Float32 inference `native_layer_norm` output/mean/rstd with affine parameters and stable Welford reduction, plus RMSNorm |
| `reduction.cpp`, `mean_dim.wgsl` | Single-dimension float32 `mean.dim`, including `keepdim`, strided views, and reduction tails |
| `scalar_binary.cpp`, `scalar_binary.wgsl` | Float32 `add/sub/mul/div` scalar, scalar-out, and scalar-in-place overloads without materializing a host scalar tensor |
| `swiglu.cpp`, `swiglu_gemv*.wgsl` | One-row float32 fused gate/up GEMV and SiLU multiplication for decode-time Llama-family MLPs, with a fixed-32 subgroup specialization and portable fallback |
| `type_conversion.cpp`, `int_to_float.wgsl`, `bool_to_long.wgsl` | Same-device `_to_copy`, including strided packed-Bool copies, packed Bool to canonical limited-int64, checked truncating float32 to limited-int64, and int32/limited-int64 to float32 conversion |
| `browser_unary.cpp` | Browser-owned float unary registrations plus restricted-Long `abs`/`neg`, without modifying the pinned vendor snapshot |
| `attention.cpp`, `sdpa.wgsl` | Float32 inference SDPA with causal alignment, broadcast additive masks, and grouped-query head mapping |

The browser acceptance tests also exercise the stock Transformers 4.46.3
RMSNorm reduction and position-ID cast patterns before composing rotary
position encoding from metadata views, negation, concatenation,
multiplication, and addition. These operators are enough for the checked
full-sequence tiny GPT block and the central float32 Llama-family primitives.
The enforced profile covers ten deterministic tiny model families: Qwen2,
Llama, Mistral, GPT-2, BERT, Phi-3, OPT, BLOOM, T5, and Gemma2. It is not enough
for arbitrary LLM repositories: generic upstream
StaticCache/sliding/beam mutations, sampling/top-k, reduced precision,
quantization, model loading, and custom operators are outside this profile.

## Arithmetic, reduction, selection, and conversion contracts

`native_layer_norm` accepts float32 ranks up to eight and any nonempty suffix
`normalized_shape`. Weight and bias are independently optional. Contiguous
inputs dispatch directly; noncontiguous inputs and affine tensors are
materialized by the GPU strided-copy path. One
workgroup owns each flattened prefix row and combines per-lane Welford
`(count, mean, M2)` states, so normalized widths need not be multiples of 256
and large common activation offsets do not suffer the cancellation of
`E[x^2] - E[x]^2`. The output is contiguous and the mean/rstd tensors retain
the normalized dimensions as size one, matching the native three-tensor ATen
schema. Empty prefix rows allocate that tuple without dispatch after validating
the affine shapes; zero-width normalized suffixes fail explicitly.

The registrations match the pinned PyTorch schemas:

```text
aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False,
               *, ScalarType? dtype=None) -> Tensor
aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
aten::all(Tensor self) -> Tensor
aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None,
               Device? device=None, bool? pin_memory=None,
               bool non_blocking=False,
               MemoryFormat? memory_format=None) -> Tensor
aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
aten::div.Scalar(Tensor self, Scalar other) -> Tensor
webgpu::fused_swiglu(Tensor input, Tensor gate_weight, Tensor up_weight,
                     Tensor? gate_bias=None, Tensor? up_bias=None) -> Tensor
webgpu::update_kv_cache_(Tensor(a!) key_cache, Tensor(b!) value_cache,
                         Tensor key_states, Tensor value_states,
                         Tensor cache_position) -> ()
```

`mean.dim` deliberately accepts one positive or negative dimension, ranks up
to eight, float32 input, and `dtype=None` or float32. It directly indexes
nonnegative input strides and storage offsets and returns the normal reduced
shape with or without the retained size-one dimension. Other reduction lists
and dtypes fail instead of redispatching through CPU.

`argmax` deliberately accepts float32 ranks one through eight and an explicit
positive or negative last dimension. It handles nonnegative strided views,
requires a nonempty reduction, preserves ATen's first-index tie and NaN
semantics, and emits ordinary 8-byte Long storage whose high word is zero.
Dimensionless, non-last-dimension, and non-float reductions fail explicitly.

`eq.Scalar` accepts float32, int32, or restricted-Long input with a compatible
bounded real scalar and emits canonical packed Bool bytes. One invocation owns
each output word, so adjacent results cannot race. `all` accepts only Bool,
reads arbitrary nonnegative strides, and reduces to a scalar Bool in one
workgroup; the empty identity is true.

`_to_copy` accepts strided layout, unpinned storage, and Preserve or Contiguous
memory format. CPU-to-WebGPU calls require matching float32, int32,
restricted-Long, or Bool dtype and route through the backend copy
implementation, which owns aligned queue upload and Long range validation;
noncontiguous CPU sources require an explicit Contiguous memory format.
Same-device WebGPU copies use the project shader's packed-byte/raw-word path,
preserving dense strides and both Long words.
Numeric conversion on WebGPU is limited to packed Bool-to-Long, checked
truncating float32-to-Long, int32-to-float32, and signed-int32-valued
Long-to-float32. Bool-to-Long reads
arbitrary nonnegative strided views and returns contiguous ordinary eight-byte
Long storage with canonical zero/one low words and zero high words. Other
limited Long values retain a canonical low i32 plus sign-extension word; the
float cast reads the low word after CPU upload and GPU producers validate that
invariant. Other devices and dtype conversions fail explicitly.

The scalar binary registrations accept real scalar values with float32 or
restricted-Long WebGPU inputs, ranks up to eight, and nonnegative strides.
Float32 supports add/subtract/multiply/divide; restricted Long supports
add/subtract/multiply, and every result must remain in the signed-int32 value
profile. `alpha` is honored for add and subtract. If an output shares its
GPUBuffer with an input, scalar and tensor binary paths first compute into a
temporary and then use the strided GPU copy kernel. This avoids WebGPU's
forbidden read-only/writable binding alias without reading data back to the
host.

`webgpu::fused_swiglu` accepts contiguous float32 inference tensors and
exactly one flattened input row. Gate and up weights must have the same
`[intermediate, hidden]` shape; biases are independently optional contiguous
vectors. Nonzero storage offsets are supported, but empty dimensions,
additional rows, mismatched devices/shapes, autograd, and non-float32 inputs
fail explicitly. A single dispatch computes both dot products and
`SiLU(gate) * up` into the intermediate tensor. Adapters with fixed-size-32
subgroups use four adjacent output rows per subgroup; all others use the
portable 64-lane workgroup shader.

`webgpu::update_kv_cache_` writes matching rank-4 key and value states into
distinct contiguous rank-4 cache allocations at a contiguous int32 or
canonical signed-int32-valued Long position vector. It fuses both writes into
one dispatch and accepts non-contiguous state views with non-negative strides.
The pinned Transformers adapter exposes only the written prefix to SDPA, so
unused cache capacity does not contribute key traffic.

The reference BMM and SDPA shaders prioritize a small auditable browser path,
not peak performance. They deliberately do not claim fusion, tensor-core use,
or optimized long-context behavior.
