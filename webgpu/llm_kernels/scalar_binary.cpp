#include "llm_common.h"
#include "long_arithmetic.h"

#include <ATen/TensorIterator.h>

#include <algorithm>
#include <cstring>

namespace pyodide_pytorch::webgpu::llm {
namespace {

enum class ScalarBinaryOp : std::uint32_t {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
};

ComputeKernel& scalar_binary_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch scalar binary",
      shaders::kScalarBinary,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

std::uint32_t float_bits(float value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

float real_scalar(const at::Scalar& scalar, const char* operation) {
  TORCH_CHECK(
      scalar.isIntegral(true) || scalar.isFloatingPoint(),
      operation,
      " requires a real scalar");
  return scalar.toFloat();
}

void validate_float_storage_span(
    const at::Tensor& tensor,
    const char* operation) {
  if (tensor.numel() == 0) {
    return;
  }
  std::uint64_t maximum_index = static_cast<std::uint64_t>(
      checked_u32(tensor.storage_offset(), "scalar-binary storage offset"));
  for (const auto dim : c10::irange(tensor.dim())) {
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        operation,
        " does not support negative strides");
    const auto size = static_cast<std::uint64_t>(tensor.size(dim));
    const auto stride = static_cast<std::uint64_t>(tensor.stride(dim));
    if (size > 0) {
      TORCH_CHECK(
          stride == 0 ||
              size - 1 <=
                  (std::numeric_limits<std::uint64_t>::max() - maximum_index) /
                      stride,
          operation,
          " storage span overflow");
      maximum_index += (size - 1) * stride;
    }
  }
  TORCH_CHECK(
      maximum_index < allocation(tensor).buffer.GetSize() / sizeof(float),
      operation,
      " view exceeds its GPUBuffer storage");
  checked_u32(
      static_cast<std::int64_t>(maximum_index),
      "scalar-binary maximum storage index");
}

std::pair<std::uint32_t, std::uint32_t> dispatch_elements(
    std::uint32_t length) {
  constexpr std::uint32_t workgroup_size = 64;
  constexpr std::uint32_t max_workgroups = 65535;
  const auto workgroups = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(length) + workgroup_size - 1) /
      workgroup_size);
  const auto x = std::min(workgroups, max_workgroups);
  const auto y = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(workgroups) + x - 1) / x);
  TORCH_CHECK(
      y <= max_workgroups, "WebGPU scalar binary dispatch is too large");
  return {x, y};
}

struct ScalarBinaryParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t input_offset;
  std::uint32_t output_offset;
  std::uint32_t scalar_bits;
  std::uint32_t alpha_bits;
  std::uint32_t operation;
  std::uint32_t dispatch_x;
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
  std::uint32_t output_strides[8];
};

static_assert(sizeof(ScalarBinaryParams) == 128);

at::Tensor& scalar_binary_out(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& output,
    ScalarBinaryOp operation,
    const char* operation_name) {
  check_inference_tensor(input, operation_name, at::kFloat);
  check_inference_tensor(output, operation_name, at::kFloat);
  TORCH_CHECK(
      input.device() == output.device(),
      operation_name,
      " requires input and output on the same WebGPU device");
  TORCH_CHECK(
      input.layout() == at::kStrided && output.layout() == at::kStrided,
      operation_name,
      " supports only strided tensors");
  TORCH_CHECK(
      input.dim() <= 8,
      operation_name,
      " supports at most 8 dimensions");

  // Use TensorIterator for standard out= resizing and overlap validation, but
  // keep the scalar as uniform data so no host scalar tensor is synthesized.
  at::TensorIteratorConfig config;
  config.set_check_mem_overlap(true);
  config.add_output(output);
  config.add_input(input);
  config.check_all_same_dtype(true);
  config.check_all_same_device(true);
  auto iterator = config.build();
  TORCH_INTERNAL_ASSERT(iterator.numel() == input.numel());
  const auto scalar_value = real_scalar(other, operation_name);
  const auto alpha_value = real_scalar(alpha, operation_name);
  if (input.numel() == 0) {
    return output;
  }

  validate_float_storage_span(input, operation_name);
  validate_float_storage_span(output, operation_name);

  // A storage buffer cannot be bound read-only and writable in one WebGPU
  // pass. Any shared allocation that passes TensorIterator's overlap checks
  // therefore computes into a temporary and uses the project strided-copy
  // kernel after the first pass.
  at::Tensor temporary;
  at::Tensor* kernel_output = &output;
  if (allocation(input).buffer.Get() == allocation(output).buffer.Get()) {
    temporary = at::empty(
        input.sizes(), input.options(), c10::MemoryFormat::Contiguous);
    kernel_output = &temporary;
  }

  ScalarBinaryParams params{};
  params.length = checked_u32(input.numel(), "scalar-binary element count");
  params.ndim = checked_u32(input.dim(), "scalar-binary rank");
  params.input_offset =
      checked_u32(input.storage_offset(), "scalar-binary input offset");
  params.output_offset = checked_u32(
      kernel_output->storage_offset(), "scalar-binary output offset");
  params.scalar_bits = float_bits(scalar_value);
  params.alpha_bits = float_bits(alpha_value);
  params.operation = static_cast<std::uint32_t>(operation);
  for (const auto dim : c10::irange(input.dim())) {
    TORCH_CHECK(
        input.stride(dim) >= 0 && kernel_output->stride(dim) >= 0,
        operation_name,
        " does not support negative strides");
    params.sizes[dim] =
        checked_u32(input.size(dim), "scalar-binary input size");
    params.input_strides[dim] =
        checked_u32(input.stride(dim), "scalar-binary input stride");
    params.output_strides[dim] = checked_u32(
        kernel_output->stride(dim), "scalar-binary output stride");
  }

  const auto [dispatch_x, dispatch_y] = dispatch_elements(params.length);
  params.dispatch_x = dispatch_x;
  auto params_buffer = make_params_buffer("scalar-binary params", params);
  dispatch(
      scalar_binary_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, *kernel_output),
       buffer_entry(2, params_buffer, sizeof(params))},
      dispatch_x,
      dispatch_y);
  if (temporary.defined()) {
    copy_strided(temporary, output);
  }
  return output;
}

at::Tensor scalar_binary(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    ScalarBinaryOp operation,
    const char* operation_name) {
  check_inference_tensor(input, operation_name, at::kFloat);
  auto output = at::empty(
      input.sizes(), input.options(), c10::MemoryFormat::Contiguous);
  scalar_binary_out(
      input, other, alpha, output, operation, operation_name);
  return output;
}

at::Tensor add_scalar(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if (input.scalar_type() == at::kLong) {
    return long_scalar(
        input,
        other,
        alpha,
        LongScalarOperation::Add,
        "WebGPU restricted Long add.Scalar");
  }
  return scalar_binary(
      input, other, alpha, ScalarBinaryOp::Add, "WebGPU add.Scalar");
}

at::Tensor& add_scalar_out(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& output) {
  if (input.scalar_type() == at::kLong) {
    return long_scalar_out(
        input,
        other,
        alpha,
        output,
        LongScalarOperation::Add,
        "WebGPU restricted Long add.Scalar");
  }
  return scalar_binary_out(
      input, other, alpha, output, ScalarBinaryOp::Add, "WebGPU add.Scalar");
}

at::Tensor& add_scalar_(
    at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  return add_scalar_out(input, other, alpha, input);
}

at::Tensor sub_scalar(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if (input.scalar_type() == at::kLong) {
    return long_scalar(
        input,
        other,
        alpha,
        LongScalarOperation::Subtract,
        "WebGPU restricted Long sub.Scalar");
  }
  return scalar_binary(
      input, other, alpha, ScalarBinaryOp::Sub, "WebGPU sub.Scalar");
}

at::Tensor& sub_scalar_out(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& output) {
  if (input.scalar_type() == at::kLong) {
    return long_scalar_out(
        input,
        other,
        alpha,
        output,
        LongScalarOperation::Subtract,
        "WebGPU restricted Long sub.Scalar");
  }
  return scalar_binary_out(
      input, other, alpha, output, ScalarBinaryOp::Sub, "WebGPU sub.Scalar");
}

at::Tensor& sub_scalar_(
    at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  return sub_scalar_out(input, other, alpha, input);
}

at::Tensor mul_scalar(
    const at::Tensor& input,
    const at::Scalar& other) {
  if (input.scalar_type() == at::kLong) {
    return long_scalar(
        input,
        other,
        1,
        LongScalarOperation::Multiply,
        "WebGPU restricted Long mul.Scalar");
  }
  return scalar_binary(
      input, other, 1, ScalarBinaryOp::Mul, "WebGPU mul.Scalar");
}

at::Tensor& mul_scalar_out(
    const at::Tensor& input,
    const at::Scalar& other,
    at::Tensor& output) {
  if (input.scalar_type() == at::kLong) {
    return long_scalar_out(
        input,
        other,
        1,
        output,
        LongScalarOperation::Multiply,
        "WebGPU restricted Long mul.Scalar");
  }
  return scalar_binary_out(
      input, other, 1, output, ScalarBinaryOp::Mul, "WebGPU mul.Scalar");
}

at::Tensor& mul_scalar_(at::Tensor& input, const at::Scalar& other) {
  return mul_scalar_out(input, other, input);
}

at::Tensor div_scalar(
    const at::Tensor& input,
    const at::Scalar& other) {
  return scalar_binary(
      input, other, 1, ScalarBinaryOp::Div, "WebGPU div.Scalar");
}

at::Tensor& div_scalar_out(
    const at::Tensor& input,
    const at::Scalar& other,
    at::Tensor& output) {
  return scalar_binary_out(
      input, other, 1, output, ScalarBinaryOp::Div, "WebGPU div.Scalar");
}

at::Tensor& div_scalar_(at::Tensor& input, const at::Scalar& other) {
  return div_scalar_out(input, other, input);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("add.Scalar", TORCH_FN(add_scalar));
  module.impl("add_.Scalar", TORCH_FN(add_scalar_));
  module.impl("add.Scalar_out", TORCH_FN(add_scalar_out));
  module.impl("sub.Scalar", TORCH_FN(sub_scalar));
  module.impl("sub_.Scalar", TORCH_FN(sub_scalar_));
  module.impl("sub.Scalar_out", TORCH_FN(sub_scalar_out));
  module.impl("mul.Scalar", TORCH_FN(mul_scalar));
  module.impl("mul_.Scalar", TORCH_FN(mul_scalar_));
  module.impl("mul.Scalar_out", TORCH_FN(mul_scalar_out));
  module.impl("div.Scalar", TORCH_FN(div_scalar));
  module.impl("div_.Scalar", TORCH_FN(div_scalar_));
  module.impl("div.Scalar_out", TORCH_FN(div_scalar_out));
}

} // namespace pyodide_pytorch::webgpu::llm
