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

The profile is inference-only and supports 32-bit storage values. Float32 is
used for model state and activations; int32 is used for token indices. The
operators reject unsupported layouts and dtypes rather than reading tensors
back through the CPU.

## Implemented vertical slice

| Source | Registered behavior |
| --- | --- |
| `copy.cpp`, `strided_copy.wgsl` | Raw 32-bit clone/contiguous materialization and concatenation for ranks up to eight |
| `embedding.cpp`, `embedding.wgsl` | Float32 embedding weights with contiguous int32 indices |
| `matmul.cpp`, `bmm.wgsl` | Strided 3-D BMM, equal-rank batched matmul, and linear composition through the existing 2-D MM kernel |
| `normalization.cpp`, `layer_norm.wgsl`, `rms_norm.wgsl` | Contiguous float32 LayerNorm and RMSNorm forward kernels |
| `attention.cpp`, `sdpa.wgsl` | Float32 inference SDPA with causal alignment, broadcast additive masks, and grouped-query head mapping |

The browser acceptance test also composes rotary position encoding from
metadata views, negation, concatenation, multiplication, and addition. These
operators are enough for the checked full-sequence tiny GPT block and the
central float32 Llama-family primitives. They are not enough for arbitrary LLM
repositories: mutable KV caches, sampling/top-k, reduced precision,
quantization, model loading, and custom operators are outside this profile.

The reference BMM and SDPA shaders prioritize a small auditable browser path,
not peak performance. They deliberately do not claim fusion, tensor-core use,
or optimized long-context behavior.
