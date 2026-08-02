#pragma once

#include <ATen/ATen.h>

namespace pyodide_pytorch::webgpu::llm {

// Called by the single patched aten::mul.Tensor registration when either
// operand is Bool. The implementation validates that both operands satisfy the
// project packed-Bool WebGPU inference contract.
at::Tensor mul_bool_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs);

} // namespace pyodide_pytorch::webgpu::llm
