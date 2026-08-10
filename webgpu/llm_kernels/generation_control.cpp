#include "llm_common.h"
#include "generation_control.h"

#include <ATen/ExpandUtils.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaximumDimensions = 8;
constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kMaximumWorkgroups = 65535;

ComputeKernel& any_bool_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch Bool any",
      shaders::kAnyBool,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& bitwise_not_bool_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch Bool bitwise not",
      shaders::kBitwiseNotBool,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& long_lt_scalar_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch restricted Long lt scalar",
      shaders::kLongLtScalar,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& mul_bool_tensor_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch Bool tensor multiply",
      shaders::kMulBoolTensor,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

void check_strided_tensor(
    const at::Tensor& tensor,
    const char* operation,
    at::ScalarType dtype) {
  check_inference_tensor(tensor, operation, dtype);
  TORCH_CHECK(
      tensor.layout() == at::kStrided,
      operation,
      " supports only strided tensors");
  TORCH_CHECK(
      tensor.dim() <= kMaximumDimensions,
      operation,
      " supports at most eight dimensions");
}

void validate_storage_span(
    const at::Tensor& tensor,
    const char* operation) {
  if (tensor.numel() == 0) {
    return;
  }

  std::uint64_t maximum_index = static_cast<std::uint64_t>(
      checked_u32(tensor.storage_offset(), "generation-control storage offset"));
  for (const auto dim : c10::irange(tensor.dim())) {
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        operation,
        " does not support negative strides");
    const auto size = static_cast<std::uint64_t>(tensor.size(dim));
    const auto stride = static_cast<std::uint64_t>(tensor.stride(dim));
    TORCH_CHECK(
        size <= std::numeric_limits<std::uint32_t>::max(),
        operation,
        " dimension does not fit uint32 WebGPU metadata");
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
      maximum_index <= std::numeric_limits<std::uint32_t>::max(),
      operation,
      " maximum storage index exceeds WGSL uint32 indexing");
  TORCH_CHECK(
      maximum_index <
          allocation(tensor).buffer.GetSize() /
              static_cast<std::uint64_t>(tensor.element_size()),
      operation,
      " view exceeds its GPUBuffer storage");
  if (tensor.scalar_type() == at::kLong) {
    TORCH_CHECK(
        maximum_index <= std::numeric_limits<std::uint32_t>::max() / 2u,
        operation,
        " Long word address exceeds WGSL uint32 indexing");
  }
}

void write_shape(
    const at::Tensor& tensor,
    std::uint32_t* sizes,
    std::uint32_t* strides,
    const char* operation) {
  for (const auto dim : c10::irange(tensor.dim())) {
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        operation,
        " does not support negative strides");
    sizes[dim] = checked_u32(tensor.size(dim), "generation-control size");
    strides[dim] =
        checked_u32(tensor.stride(dim), "generation-control stride");
  }
}

struct DispatchShape {
  std::uint32_t x;
  std::uint32_t y;
};

DispatchShape dispatch_words(
    std::uint32_t elements,
    const char* operation) {
  TORCH_INTERNAL_ASSERT(elements > 0);
  const auto words = (static_cast<std::uint64_t>(elements) + 3u) / 4u;
  const auto groups = (words + kWorkgroupSize - 1u) / kWorkgroupSize;
  const auto x = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(groups, kMaximumWorkgroups));
  const auto y = static_cast<std::uint32_t>((groups + x - 1u) / x);
  TORCH_CHECK(
      y <= kMaximumWorkgroups,
      operation,
      " dispatch exceeds WebGPU limits");
  return {x, y};
}

struct AnyBoolParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t input_offset;
  std::uint32_t padding;
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
};

static_assert(sizeof(AnyBoolParams) == 80);

at::Tensor any_bool(const at::Tensor& input) {
  constexpr const char* operation = "WebGPU Bool any";
  check_strided_tensor(input, operation, at::kBool);
  auto output = at::empty({}, input.options());
  if (input.numel() == 0) {
    output.fill_(false);
    return output;
  }
  validate_storage_span(input, operation);
  validate_storage_span(output, operation);

  AnyBoolParams params{};
  params.length = checked_u32(input.numel(), "Bool any element count");
  params.ndim = checked_u32(input.dim(), "Bool any input rank");
  params.input_offset =
      checked_u32(input.storage_offset(), "Bool any input offset");
  write_shape(
      input, params.sizes, params.input_strides, operation);

  auto params_buffer = make_params_buffer("Bool any params", params);
  dispatch(
      any_bool_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      1);
  return output;
}

struct BitwiseNotBoolParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t input_offset;
  std::uint32_t dispatch_x;
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
};

static_assert(sizeof(BitwiseNotBoolParams) == 80);

at::Tensor bitwise_not_bool(const at::Tensor& input) {
  constexpr const char* operation = "WebGPU Bool bitwise_not";
  check_strided_tensor(input, operation, at::kBool);
  auto output = at::empty(
      input.sizes(),
      input.options(),
      c10::MemoryFormat::Contiguous);
  if (input.numel() == 0) {
    return output;
  }
  validate_storage_span(input, operation);
  validate_storage_span(output, operation);

  BitwiseNotBoolParams params{};
  params.length = checked_u32(input.numel(), "Bool bitwise_not element count");
  params.ndim = checked_u32(input.dim(), "Bool bitwise_not input rank");
  params.input_offset =
      checked_u32(input.storage_offset(), "Bool bitwise_not input offset");
  write_shape(
      input, params.sizes, params.input_strides, operation);
  const auto shape = dispatch_words(params.length, operation);
  params.dispatch_x = shape.x;

  auto params_buffer = make_params_buffer("Bool bitwise_not params", params);
  dispatch(
      bitwise_not_bool_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

struct MulBoolTensorParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t lhs_ndim;
  std::uint32_t rhs_ndim;
  std::uint32_t lhs_offset;
  std::uint32_t rhs_offset;
  std::uint32_t dispatch_x;
  std::uint32_t padding;
  std::uint32_t output_sizes[8];
  std::uint32_t lhs_sizes[8];
  std::uint32_t lhs_strides[8];
  std::uint32_t rhs_sizes[8];
  std::uint32_t rhs_strides[8];
};

static_assert(sizeof(MulBoolTensorParams) == 192);

at::Tensor mul_bool_tensor_impl(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  constexpr const char* operation = "WebGPU Bool mul.Tensor";
  check_strided_tensor(lhs, operation, at::kBool);
  check_strided_tensor(rhs, operation, at::kBool);
  TORCH_CHECK(
      lhs.device() == rhs.device(),
      operation,
      " requires both inputs on the same WebGPU device");

  auto output_shape = at::infer_size(lhs.sizes(), rhs.sizes());
  TORCH_CHECK(
      output_shape.size() <= kMaximumDimensions,
      operation,
      " broadcast result has more than eight dimensions");
  auto output = at::empty(
      output_shape,
      lhs.options(),
      c10::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }
  validate_storage_span(lhs, operation);
  validate_storage_span(rhs, operation);
  validate_storage_span(output, operation);

  MulBoolTensorParams params{};
  params.length = checked_u32(output.numel(), "Bool mul.Tensor output length");
  params.ndim = checked_u32(output.dim(), "Bool mul.Tensor output rank");
  params.lhs_ndim = checked_u32(lhs.dim(), "Bool mul.Tensor lhs rank");
  params.rhs_ndim = checked_u32(rhs.dim(), "Bool mul.Tensor rhs rank");
  params.lhs_offset =
      checked_u32(lhs.storage_offset(), "Bool mul.Tensor lhs offset");
  params.rhs_offset =
      checked_u32(rhs.storage_offset(), "Bool mul.Tensor rhs offset");
  for (const auto dim : c10::irange(output.dim())) {
    params.output_sizes[dim] =
        checked_u32(output.size(dim), "Bool mul.Tensor output size");
  }
  write_shape(lhs, params.lhs_sizes, params.lhs_strides, operation);
  write_shape(rhs, params.rhs_sizes, params.rhs_strides, operation);
  const auto shape = dispatch_words(params.length, operation);
  params.dispatch_x = shape.x;

  auto params_buffer = make_params_buffer("Bool mul.Tensor params", params);
  dispatch(
      mul_bool_tensor_kernel(),
      {tensor_entry(0, lhs),
       tensor_entry(1, rhs),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

std::int32_t scalar_to_i32(
    const at::Scalar& scalar,
    const char* operation) {
  std::int64_t value;
  if (scalar.isIntegral(true)) {
    value = scalar.toLong();
  } else if (scalar.isFloatingPoint()) {
    const auto floating = scalar.toDouble();
    TORCH_CHECK(
        std::isfinite(floating) && std::trunc(floating) == floating,
        operation,
        " supports only integral scalar values");
    TORCH_CHECK(
        floating >=
                static_cast<double>(std::numeric_limits<std::int32_t>::min()) &&
            floating <=
                static_cast<double>(std::numeric_limits<std::int32_t>::max()),
        operation,
        " scalar is outside the signed-int32 WebGPU profile");
    value = static_cast<std::int64_t>(floating);
  } else {
    TORCH_CHECK(false, operation, " requires a real scalar");
  }
  TORCH_CHECK(
      value >= std::numeric_limits<std::int32_t>::min() &&
          value <= std::numeric_limits<std::int32_t>::max(),
      operation,
      " scalar is outside the signed-int32 WebGPU profile");
  return static_cast<std::int32_t>(value);
}

std::uint32_t raw_bits(std::int32_t value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct LongLtScalarParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t input_offset;
  std::uint32_t scalar_bits;
  std::uint32_t dispatch_x;
  std::uint32_t comparison;
  std::uint32_t padding[2];
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
};

static_assert(sizeof(LongLtScalarParams) == 96);

at::Tensor long_compare_scalar(
    const at::Tensor& input,
    const at::Scalar& other,
    bool greater) {
  const auto* operation = greater
      ? "WebGPU restricted Long gt.Scalar"
      : "WebGPU restricted Long lt.Scalar";
  check_strided_tensor(input, operation, at::kLong);
  const auto scalar = scalar_to_i32(other, operation);
  auto output = at::empty(
      input.sizes(),
      input.options().dtype(at::kBool),
      c10::MemoryFormat::Contiguous);
  if (input.numel() == 0) {
    return output;
  }
  validate_storage_span(input, operation);
  validate_storage_span(output, operation);

  LongLtScalarParams params{};
  params.length = checked_u32(input.numel(), "Long lt.Scalar element count");
  params.ndim = checked_u32(input.dim(), "Long lt.Scalar input rank");
  params.input_offset =
      checked_u32(input.storage_offset(), "Long comparison input offset");
  params.scalar_bits = raw_bits(scalar);
  params.comparison = greater ? 1u : 0u;
  write_shape(
      input, params.sizes, params.input_strides, operation);
  const auto shape = dispatch_words(params.length, operation);
  params.dispatch_x = shape.x;

  auto params_buffer = make_params_buffer("Long lt.Scalar params", params);
  dispatch(
      long_lt_scalar_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

at::Tensor long_lt_scalar(
    const at::Tensor& input,
    const at::Scalar& other) {
  return long_compare_scalar(input, other, false);
}

at::Tensor long_gt_scalar(
    const at::Tensor& input,
    const at::Scalar& other) {
  return long_compare_scalar(input, other, true);
}

} // namespace

at::Tensor mul_bool_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  return mul_bool_tensor_impl(lhs, rhs);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("any", TORCH_FN(any_bool));
  module.impl("bitwise_not", TORCH_FN(bitwise_not_bool));
  module.impl("lt.Scalar", TORCH_FN(long_lt_scalar));
  module.impl("gt.Scalar", TORCH_FN(long_gt_scalar));
}

} // namespace pyodide_pytorch::webgpu::llm
