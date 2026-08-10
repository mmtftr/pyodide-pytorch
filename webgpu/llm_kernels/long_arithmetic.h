#pragma once

#include <ATen/ATen.h>

namespace pyodide_pytorch::webgpu::llm {

enum class LongScalarOperation {
  Add,
  Subtract,
  Multiply,
};

at::Tensor add_long_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Scalar& alpha = 1);

at::Tensor sub_long_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Scalar& alpha = 1);

at::Tensor mul_long_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs);

at::Tensor mul_float_integer_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs);

at::Tensor abs_long_tensor(const at::Tensor& input);

at::Tensor neg_long_tensor(const at::Tensor& input);

at::Tensor& abs_long_out(const at::Tensor& input, at::Tensor& output);

at::Tensor& neg_long_out(const at::Tensor& input, at::Tensor& output);

at::Tensor long_scalar(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    LongScalarOperation operation,
    const char* operation_name);

at::Tensor& long_scalar_out(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& output,
    LongScalarOperation operation,
    const char* operation_name);

} // namespace pyodide_pytorch::webgpu::llm
