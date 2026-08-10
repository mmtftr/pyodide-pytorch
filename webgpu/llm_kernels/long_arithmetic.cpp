#include "long_arithmetic.h"

#include "llm_common.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/sub_ops.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaximumDimensions = 8;
constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kMaximumWorkgroups = 65535;

enum class Operation : std::uint32_t {
  Add = 0,
  Subtract = 1,
  Multiply = 2,
  Minimum = 3,
  ReverseSubtract = 4,
  Negate = 5,
  Absolute = 6,
};

struct DispatchShape {
  std::uint32_t x;
  std::uint32_t y;
};

DispatchShape dispatch_shape(std::uint32_t elements, const char* operation) {
  TORCH_INTERNAL_ASSERT(elements > 0);
  const auto groups =
      (static_cast<std::uint64_t>(elements) + kWorkgroupSize - 1) /
      kWorkgroupSize;
  const auto x = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(groups, kMaximumWorkgroups));
  const auto y = static_cast<std::uint32_t>((groups + x - 1) / x);
  TORCH_CHECK(y <= kMaximumWorkgroups, operation, " dispatch exceeds WebGPU limits");
  return {x, y};
}

std::int32_t scalar_i32(const at::Scalar& value, const char* operation) {
  std::int64_t integer;
  if (value.isIntegral(true)) {
    integer = value.toLong();
  } else if (value.isFloatingPoint()) {
    const auto floating = value.toDouble();
    TORCH_CHECK(
        std::isfinite(floating) && std::trunc(floating) == floating,
        operation,
        " requires an integral scalar");
    TORCH_CHECK(
        floating >= static_cast<double>(std::numeric_limits<std::int32_t>::min()) &&
            floating <= static_cast<double>(std::numeric_limits<std::int32_t>::max()),
        operation,
        " scalar is outside the signed-int32 WebGPU profile");
    integer = static_cast<std::int64_t>(floating);
  } else {
    TORCH_CHECK(false, operation, " requires a real scalar");
  }
  TORCH_CHECK(
      integer >= std::numeric_limits<std::int32_t>::min() &&
          integer <= std::numeric_limits<std::int32_t>::max(),
      operation,
      " scalar is outside the signed-int32 WebGPU profile");
  return static_cast<std::int32_t>(integer);
}

std::uint32_t raw_bits(std::int32_t value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

void check_long(const at::Tensor& tensor, const char* operation) {
  check_inference_tensor(tensor, operation, at::kLong);
  TORCH_CHECK(tensor.layout() == at::kStrided, operation, " supports only strided tensors");
  TORCH_CHECK(tensor.dim() <= kMaximumDimensions, operation, " supports at most eight dimensions");
}

void validate_span(const at::Tensor& tensor, const char* operation) {
  if (tensor.numel() == 0) {
    return;
  }
  std::uint64_t maximum = static_cast<std::uint64_t>(
      checked_u32(tensor.storage_offset(), "restricted-Long storage offset"));
  for (const auto dim : c10::irange(tensor.dim())) {
    TORCH_CHECK(tensor.stride(dim) >= 0, operation, " does not support negative strides");
    const auto size = static_cast<std::uint64_t>(tensor.size(dim));
    const auto stride = static_cast<std::uint64_t>(tensor.stride(dim));
    if (size > 0) {
      TORCH_CHECK(
          stride == 0 ||
              size - 1 <=
                  (std::numeric_limits<std::uint64_t>::max() - maximum) / stride,
          operation,
          " storage span overflow");
      maximum += (size - 1) * stride;
    }
  }
  const auto element_words = tensor.element_size() / sizeof(std::uint32_t);
  TORCH_CHECK(
      element_words > 0 &&
          maximum <=
              std::numeric_limits<std::uint32_t>::max() / element_words,
      operation,
      " word address exceeds WGSL uint32 indexing");
  TORCH_CHECK(
      maximum < allocation(tensor).buffer.GetSize() / tensor.element_size(),
      operation,
      " view exceeds its GPUBuffer storage");
}

void write_shape(
    const at::Tensor& tensor,
    std::uint32_t* sizes,
    std::uint32_t* strides,
    const char* operation) {
  for (const auto dim : c10::irange(tensor.dim())) {
    sizes[dim] = checked_u32(tensor.size(dim), "restricted-Long dimension");
    TORCH_CHECK(tensor.stride(dim) >= 0, operation, " does not support negative strides");
    strides[dim] = checked_u32(tensor.stride(dim), "restricted-Long stride");
  }
}

ComputeKernel& arithmetic_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch restricted Long arithmetic",
      shaders::kLongArithmetic,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct ArithmeticParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t lhs_ndim;
  std::uint32_t rhs_ndim;
  std::uint32_t lhs_offset;
  std::uint32_t rhs_offset;
  std::uint32_t operation;
  std::uint32_t alpha_low;
  std::uint32_t scalar_low;
  std::uint32_t scalar_high;
  std::uint32_t rhs_is_scalar;
  std::uint32_t dispatch_x;
  std::uint32_t output_sizes[8];
  std::uint32_t lhs_sizes[8];
  std::uint32_t lhs_strides[8];
  std::uint32_t rhs_sizes[8];
  std::uint32_t rhs_strides[8];
};

static_assert(sizeof(ArithmeticParams) == 208);

at::Tensor arithmetic(
    const at::Tensor& lhs,
    const at::Tensor* rhs,
    const at::Scalar& scalar,
    const at::Scalar& alpha,
    Operation operation,
    const char* operation_name) {
  check_long(lhs, operation_name);
  if (rhs != nullptr) {
    check_long(*rhs, operation_name);
    TORCH_CHECK(lhs.device() == rhs->device(), operation_name, " requires one WebGPU device");
  }
  const auto output_shape = rhs == nullptr
      ? lhs.sizes().vec()
      : at::infer_size(lhs.sizes(), rhs->sizes());
  TORCH_CHECK(
      output_shape.size() <= kMaximumDimensions,
      operation_name,
      " result rank exceeds eight");
  auto output = at::empty(
      output_shape,
      lhs.options(),
      c10::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }
  validate_span(lhs, operation_name);
  if (rhs != nullptr) {
    validate_span(*rhs, operation_name);
  }
  validate_span(output, operation_name);

  ArithmeticParams params{};
  params.length = checked_u32(output.numel(), "restricted-Long output length");
  params.ndim = checked_u32(output.dim(), "restricted-Long output rank");
  params.lhs_ndim = checked_u32(lhs.dim(), "restricted-Long lhs rank");
  params.rhs_ndim = rhs == nullptr ? 0 : checked_u32(rhs->dim(), "restricted-Long rhs rank");
  params.lhs_offset = checked_u32(lhs.storage_offset(), "restricted-Long lhs offset");
  params.rhs_offset = rhs == nullptr
      ? 0
      : checked_u32(rhs->storage_offset(), "restricted-Long rhs offset");
  params.operation = static_cast<std::uint32_t>(operation);
  params.alpha_low = raw_bits(scalar_i32(alpha, operation_name));
  const auto scalar_value = scalar_i32(scalar, operation_name);
  params.scalar_low = raw_bits(scalar_value);
  params.scalar_high = scalar_value < 0 ? 0xffffffffu : 0u;
  params.rhs_is_scalar = rhs == nullptr ? 1u : 0u;
  for (const auto dim : c10::irange(output.dim())) {
    params.output_sizes[dim] = checked_u32(output.size(dim), "restricted-Long output size");
  }
  write_shape(lhs, params.lhs_sizes, params.lhs_strides, operation_name);
  if (rhs != nullptr) {
    write_shape(*rhs, params.rhs_sizes, params.rhs_strides, operation_name);
  }
  const auto shape = dispatch_shape(params.length, operation_name);
  params.dispatch_x = shape.x;
  const auto& rhs_binding = rhs == nullptr ? lhs : *rhs;
  auto params_buffer = make_params_buffer("restricted-Long arithmetic params", params);
  dispatch(
      arithmetic_kernel(),
      {tensor_entry(0, lhs),
       tensor_entry(1, rhs_binding),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

at::Tensor& copy_result(const at::Tensor& result, at::Tensor& output, const char* operation) {
  check_long(output, operation);
  TORCH_CHECK(result.device() == output.device(), operation, " requires one WebGPU device");
  TORCH_CHECK(result.sizes() == output.sizes(), operation, " output shape mismatch");
  copy_strided(result, output);
  return output;
}

at::Tensor minimum_long(const at::Tensor& lhs, const at::Tensor& rhs) {
  if (lhs.scalar_type() == at::kFloat && rhs.scalar_type() == at::kFloat) {
    return at::where(at::gt(lhs, rhs), rhs, lhs);
  }
  return arithmetic(lhs, &rhs, 0, 1, Operation::Minimum, "WebGPU restricted Long minimum");
}

at::Tensor& minimum_long_out(const at::Tensor& lhs, const at::Tensor& rhs, at::Tensor& output) {
  auto result = minimum_long(lhs, rhs);
  if (result.scalar_type() == at::kLong) {
    return copy_result(result, output, "WebGPU restricted Long minimum.out");
  }
  check_inference_tensor(output, "WebGPU float32 minimum.out", at::kFloat);
  copy_strided(result, output);
  return output;
}

at::Tensor reverse_subtract_long(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  if (input.scalar_type() == at::kFloat) {
    return at::add(
        at::mul(input, at::Scalar(-alpha.toDouble())), other);
  }
  return arithmetic(
      input,
      nullptr,
      other,
      alpha,
      Operation::ReverseSubtract,
      "WebGPU restricted Long rsub.Scalar");
}

at::Tensor unary_long(const at::Tensor& input, Operation operation, const char* name) {
  return arithmetic(input, nullptr, 0, 1, operation, name);
}

at::Tensor& add_tensor_inplace(
    at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Scalar& alpha) {
  if (lhs.scalar_type() != at::kLong || rhs.scalar_type() != at::kLong) {
    return at::_ops::add_out::call(lhs, rhs, alpha, lhs);
  }
  return copy_result(add_long_tensor(lhs, rhs, alpha), lhs, "WebGPU restricted Long add_.Tensor");
}

at::Tensor& sub_tensor_inplace(
    at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Scalar& alpha) {
  if (lhs.scalar_type() != at::kLong || rhs.scalar_type() != at::kLong) {
    return at::_ops::sub_out::call(lhs, rhs, alpha, lhs);
  }
  return copy_result(sub_long_tensor(lhs, rhs, alpha), lhs, "WebGPU restricted Long sub_.Tensor");
}

ComputeKernel& mixed_pow_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch mixed integer-exponent pow",
      shaders::kMixedPow,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct PowParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t base_ndim;
  std::uint32_t exponent_ndim;
  std::uint32_t base_offset;
  std::uint32_t exponent_offset;
  std::uint32_t exponent_words;
  std::uint32_t dispatch_x;
  std::uint32_t output_sizes[8];
  std::uint32_t base_sizes[8];
  std::uint32_t base_strides[8];
  std::uint32_t exponent_sizes[8];
  std::uint32_t exponent_strides[8];
};

static_assert(sizeof(PowParams) == 192);

at::Tensor mixed_pow(const at::Tensor& base, const at::Tensor& exponent) {
  constexpr const char* operation = "WebGPU mixed integer-exponent pow.Tensor_Tensor";
  check_inference_tensor(base, operation, at::kFloat);
  TORCH_CHECK(
      exponent.scalar_type() == at::kFloat || exponent.scalar_type() == at::kInt ||
          exponent.scalar_type() == at::kLong,
      operation,
      " requires a float32, int32, or restricted-Long exponent");
  check_inference_tensor(exponent, operation, exponent.scalar_type());
  TORCH_CHECK(base.device() == exponent.device(), operation, " requires one WebGPU device");
  TORCH_CHECK(
      base.dim() <= 8 && exponent.dim() <= 8,
      operation,
      " supports at most eight dimensions");
  const auto output_shape = at::infer_size(base.sizes(), exponent.sizes());
  auto output = at::empty(output_shape, base.options(), c10::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }
  validate_span(base, operation);
  validate_span(exponent, operation);
  validate_span(output, operation);

  PowParams params{};
  params.length = checked_u32(output.numel(), "mixed pow output length");
  params.ndim = checked_u32(output.dim(), "mixed pow output rank");
  params.base_ndim = checked_u32(base.dim(), "mixed pow base rank");
  params.exponent_ndim = checked_u32(exponent.dim(), "mixed pow exponent rank");
  params.base_offset = checked_u32(base.storage_offset(), "mixed pow base offset");
  params.exponent_offset = checked_u32(exponent.storage_offset(), "mixed pow exponent offset");
  params.exponent_words = exponent.scalar_type() == at::kFloat
      ? 0u
      : (exponent.scalar_type() == at::kLong ? 2u : 1u);
  for (const auto dim : c10::irange(output.dim())) {
    params.output_sizes[dim] = checked_u32(output.size(dim), "mixed pow output size");
  }
  write_shape(base, params.base_sizes, params.base_strides, operation);
  write_shape(exponent, params.exponent_sizes, params.exponent_strides, operation);
  const auto shape = dispatch_shape(params.length, operation);
  params.dispatch_x = shape.x;
  auto params_buffer = make_params_buffer("mixed pow params", params);
  dispatch(
      mixed_pow_kernel(),
      {tensor_entry(0, base),
       tensor_entry(1, exponent),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

at::Tensor& mixed_pow_out(
    const at::Tensor& base,
    const at::Tensor& exponent,
    at::Tensor& output) {
  auto result = mixed_pow(base, exponent);
  TORCH_CHECK(output.scalar_type() == at::kFloat, "WebGPU mixed pow output must be float32");
  copy_strided(result, output);
  return output;
}

} // namespace

at::Tensor add_long_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Scalar& alpha) {
  return arithmetic(lhs, &rhs, 0, alpha, Operation::Add, "WebGPU restricted Long add.Tensor");
}

at::Tensor sub_long_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const at::Scalar& alpha) {
  return arithmetic(lhs, &rhs, 0, alpha, Operation::Subtract, "WebGPU restricted Long sub.Tensor");
}

at::Tensor mul_long_tensor(const at::Tensor& lhs, const at::Tensor& rhs) {
  return arithmetic(lhs, &rhs, 0, 1, Operation::Multiply, "WebGPU restricted Long mul.Tensor");
}

at::Tensor mul_float_integer_tensor(const at::Tensor& lhs, const at::Tensor& rhs) {
  constexpr const char* operation = "WebGPU mixed Float/integer mul.Tensor";
  const at::Tensor* floating = &lhs;
  const at::Tensor* integer = &rhs;
  if (lhs.scalar_type() != at::kFloat) {
    floating = &rhs;
    integer = &lhs;
  }
  TORCH_CHECK(floating->scalar_type() == at::kFloat, operation, " requires one float32 input");
  TORCH_CHECK(
      integer->scalar_type() == at::kInt || integer->scalar_type() == at::kLong,
      operation,
      " requires one int32 or restricted-Long input");
  TORCH_CHECK(floating->device() == integer->device(), operation, " requires one WebGPU device");
  auto converted = integer->to(at::kFloat);
  return at::mul(*floating, converted);
}

at::Tensor abs_long_tensor(const at::Tensor& input) {
  return unary_long(
      input, Operation::Absolute, "WebGPU restricted Long abs");
}

at::Tensor neg_long_tensor(const at::Tensor& input) {
  return unary_long(
      input, Operation::Negate, "WebGPU restricted Long neg");
}

at::Tensor& abs_long_out(const at::Tensor& input, at::Tensor& output) {
  return copy_result(
      abs_long_tensor(input), output, "WebGPU restricted Long abs.out");
}

at::Tensor& neg_long_out(const at::Tensor& input, at::Tensor& output) {
  return copy_result(
      neg_long_tensor(input), output, "WebGPU restricted Long neg.out");
}

at::Tensor long_scalar(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    LongScalarOperation operation,
    const char* operation_name) {
  const auto kernel_operation = operation == LongScalarOperation::Add
      ? Operation::Add
      : (operation == LongScalarOperation::Subtract
             ? Operation::Subtract
             : Operation::Multiply);
  return arithmetic(input, nullptr, other, alpha, kernel_operation, operation_name);
}

at::Tensor& long_scalar_out(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha,
    at::Tensor& output,
    LongScalarOperation operation,
    const char* operation_name) {
  return copy_result(
      long_scalar(input, other, alpha, operation, operation_name),
      output,
      operation_name);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("add_.Tensor", TORCH_FN(add_tensor_inplace));
  module.impl("sub_.Tensor", TORCH_FN(sub_tensor_inplace));
  module.impl("minimum", TORCH_FN(minimum_long));
  module.impl("minimum.out", TORCH_FN(minimum_long_out));
  module.impl("rsub.Scalar", TORCH_FN(reverse_subtract_long));
  module.impl("pow.Tensor_Tensor", TORCH_FN(mixed_pow));
  module.impl("pow.Tensor_Tensor_out", TORCH_FN(mixed_pow_out));
}

} // namespace pyodide_pytorch::webgpu::llm
