#include "llm_common.h"
#include "long_arithmetic.h"

#include "unary.h"

#include <ATen/TensorIterator.h>

namespace pyodide_pytorch::webgpu::llm {
namespace {

template <torch_webgpu::ops::UnaryOp Operation>
at::Tensor float_unary(const at::Tensor& input, const char* operation_name) {
  check_inference_tensor(input, operation_name, at::kFloat);
  auto output = at::empty_like(input);
  at::TensorIteratorConfig config;
  config.set_check_mem_overlap(true);
  config.add_output(output);
  config.add_input(input);
  config.check_all_same_dtype(true);
  config.check_all_same_device(true);
  auto iterator = config.build();
  if (iterator.numel() != 0) {
    torch_webgpu::ops::unary_kernel<Operation>(iterator);
  }
  return output;
}

template <torch_webgpu::ops::UnaryOp Operation>
at::Tensor& float_unary_out(
    const at::Tensor& input,
    at::Tensor& output,
    const char* operation_name) {
  check_inference_tensor(input, operation_name, at::kFloat);
  check_inference_tensor(output, operation_name, at::kFloat);
  if (input.numel() != 0 &&
      allocation(input).buffer.Get() == allocation(output).buffer.Get()) {
    auto temporary = float_unary<Operation>(input, operation_name);
    copy_strided(temporary, output);
    return output;
  }
  at::TensorIteratorConfig config;
  config.set_check_mem_overlap(true);
  config.add_output(output);
  config.add_input(input);
  config.check_all_same_dtype(true);
  config.check_all_same_device(true);
  auto iterator = config.build();
  if (iterator.numel() != 0) {
    torch_webgpu::ops::unary_kernel<Operation>(iterator);
  }
  return output;
}

#define WEBGPU_FLOAT_UNARY(name, enum_name)                                  \
  at::Tensor name##_impl(const at::Tensor& input) {                          \
    return float_unary<torch_webgpu::ops::UnaryOp::enum_name>(               \
        input, "WebGPU " #name);                                             \
  }                                                                           \
  at::Tensor& name##_out(const at::Tensor& input, at::Tensor& output) {       \
    return float_unary_out<torch_webgpu::ops::UnaryOp::enum_name>(            \
        input, output, "WebGPU " #name ".out");                              \
  }

WEBGPU_FLOAT_UNARY(cos, Cos)
WEBGPU_FLOAT_UNARY(sin, Sin)
WEBGPU_FLOAT_UNARY(tanh, Tanh)
WEBGPU_FLOAT_UNARY(exp, Exp)
WEBGPU_FLOAT_UNARY(rsqrt, Rsqrt)
WEBGPU_FLOAT_UNARY(log, Log)

#undef WEBGPU_FLOAT_UNARY

at::Tensor abs_impl(const at::Tensor& input) {
  if (input.scalar_type() == at::kLong) {
    return abs_long_tensor(input);
  }
  return float_unary<torch_webgpu::ops::UnaryOp::Abs>(input, "WebGPU abs");
}

at::Tensor& abs_out(const at::Tensor& input, at::Tensor& output) {
  if (input.scalar_type() == at::kLong) {
    return abs_long_out(input, output);
  }
  return float_unary_out<torch_webgpu::ops::UnaryOp::Abs>(
      input, output, "WebGPU abs.out");
}

at::Tensor neg_impl(const at::Tensor& input) {
  if (input.scalar_type() == at::kLong) {
    return neg_long_tensor(input);
  }
  return float_unary<torch_webgpu::ops::UnaryOp::Neg>(input, "WebGPU neg");
}

at::Tensor& neg_out(const at::Tensor& input, at::Tensor& output) {
  if (input.scalar_type() == at::kLong) {
    return neg_long_out(input, output);
  }
  return float_unary_out<torch_webgpu::ops::UnaryOp::Neg>(
      input, output, "WebGPU neg.out");
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("cos", TORCH_FN(cos_impl));
  module.impl("cos.out", TORCH_FN(cos_out));
  module.impl("sin", TORCH_FN(sin_impl));
  module.impl("sin.out", TORCH_FN(sin_out));
  module.impl("tanh", TORCH_FN(tanh_impl));
  module.impl("tanh.out", TORCH_FN(tanh_out));
  module.impl("exp", TORCH_FN(exp_impl));
  module.impl("exp.out", TORCH_FN(exp_out));
  module.impl("abs", TORCH_FN(abs_impl));
  module.impl("abs.out", TORCH_FN(abs_out));
  module.impl("rsqrt", TORCH_FN(rsqrt_impl));
  module.impl("rsqrt.out", TORCH_FN(rsqrt_out));
  module.impl("neg", TORCH_FN(neg_impl));
  module.impl("neg.out", TORCH_FN(neg_out));
  module.impl("log", TORCH_FN(log_impl));
  module.impl("log.out", TORCH_FN(log_out));
}

} // namespace pyodide_pytorch::webgpu::llm
