#include "llm_common.h"

#include <ATen/ExpandUtils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& eq_scalar_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch eq scalar",
      shaders::kEqScalar,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& ne_tensor_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch integer ne tensor",
      shaders::kNeTensor,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& all_bool_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch bool all",
      shaders::kAllBool,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

std::uint32_t raw_bits(std::int32_t value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint32_t raw_bits(float value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::int32_t comparison_i32(
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
        " supports only integral scalar values for integer tensors");
    TORCH_CHECK(
        floating >= static_cast<double>(std::numeric_limits<std::int32_t>::min()) &&
            floating <= static_cast<double>(std::numeric_limits<std::int32_t>::max()),
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

void validate_storage_span(
    const at::Tensor& tensor,
    const char* operation) {
  if (tensor.numel() == 0) {
    return;
  }
  std::uint64_t maximum_index =
      static_cast<std::uint64_t>(tensor.storage_offset());
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
  checked_u32(
      static_cast<std::int64_t>(maximum_index),
      "Bool/comparison maximum storage index");
  const auto element_bytes = static_cast<std::uint64_t>(tensor.element_size());
  TORCH_CHECK(
      maximum_index <
          allocation(tensor).buffer.GetSize() / element_bytes,
      operation,
      " view exceeds its GPUBuffer storage");
  if (tensor.scalar_type() == at::kLong) {
    TORCH_CHECK(
        maximum_index <= std::numeric_limits<std::uint32_t>::max() / 2u,
        operation,
        " Long word address exceeds WGSL uint32 indexing");
  }
}

std::pair<std::uint32_t, std::uint32_t> dispatch_words(
    std::uint32_t elements) {
  constexpr std::uint32_t workgroup_size = 64;
  constexpr std::uint32_t max_workgroups = 65535;
  const auto words = (static_cast<std::uint64_t>(elements) + 3) / 4;
  const auto groups = (words + workgroup_size - 1) / workgroup_size;
  const auto x = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(groups, max_workgroups));
  const auto y = static_cast<std::uint32_t>((groups + x - 1) / x);
  TORCH_CHECK(y <= max_workgroups, "WebGPU comparison dispatch is too large");
  return {x, y};
}

struct EqScalarParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t input_offset;
  std::uint32_t input_kind;
  std::uint32_t scalar_low;
  std::uint32_t scalar_high;
  std::uint32_t dispatch_x;
  std::uint32_t padding;
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
};

static_assert(sizeof(EqScalarParams) == 96);

at::Tensor eq_scalar(
    const at::Tensor& input,
    const at::Scalar& other) {
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kInt ||
          input.scalar_type() == at::kLong,
      "WebGPU eq.Scalar supports only torch.float32, torch.int32, and "
      "signed-int32-valued torch.int64 input");
  check_inference_tensor(input, "WebGPU eq.Scalar input", input.scalar_type());
  TORCH_CHECK(
      input.layout() == at::kStrided,
      "WebGPU eq.Scalar supports only strided tensors");
  TORCH_CHECK(
      input.dim() <= 8,
      "WebGPU eq.Scalar supports at most 8 dimensions");

  auto output = at::empty(
      input.sizes(),
      input.options().dtype(at::kBool),
      c10::MemoryFormat::Contiguous);
  if (input.numel() == 0) {
    return output;
  }
  validate_storage_span(input, "WebGPU eq.Scalar input");
  validate_storage_span(output, "WebGPU eq.Scalar output");

  EqScalarParams params{};
  params.length = checked_u32(input.numel(), "eq.Scalar element count");
  params.ndim = checked_u32(input.dim(), "eq.Scalar rank");
  params.input_offset =
      checked_u32(input.storage_offset(), "eq.Scalar input offset");
  if (input.scalar_type() == at::kFloat) {
    TORCH_CHECK(
        other.isIntegral(true) || other.isFloatingPoint(),
        "WebGPU eq.Scalar requires a real scalar");
    params.input_kind = 0;
    params.scalar_low = raw_bits(other.toFloat());
  } else {
    const auto integer = comparison_i32(other, "WebGPU eq.Scalar");
    params.input_kind = input.scalar_type() == at::kInt ? 1 : 2;
    params.scalar_low = raw_bits(integer);
    params.scalar_high = integer < 0 ? 0xffffffffu : 0u;
  }
  for (const auto dim : c10::irange(input.dim())) {
    params.sizes[dim] = checked_u32(input.size(dim), "eq.Scalar input size");
    params.input_strides[dim] =
        checked_u32(input.stride(dim), "eq.Scalar input stride");
  }

  const auto [dispatch_x, dispatch_y] = dispatch_words(params.length);
  params.dispatch_x = dispatch_x;
  auto params_buffer = make_params_buffer("eq.Scalar params", params);
  dispatch(
      eq_scalar_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      dispatch_x,
      dispatch_y);
  return output;
}

struct NeTensorParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t lhs_ndim;
  std::uint32_t rhs_ndim;
  std::uint32_t lhs_offset;
  std::uint32_t rhs_offset;
  std::uint32_t input_kinds;
  std::uint32_t dispatch_x;
  std::uint32_t output_sizes[8];
  std::uint32_t lhs_sizes[8];
  std::uint32_t lhs_strides[8];
  std::uint32_t rhs_sizes[8];
  std::uint32_t rhs_strides[8];
};

static_assert(sizeof(NeTensorParams) == 192);

void check_ne_input(
    const at::Tensor& input,
    const char* operation) {
  TORCH_CHECK(
      input.scalar_type() == at::kInt || input.scalar_type() == at::kLong,
      operation,
      " supports only torch.int32 and signed-int32-valued torch.int64");
  check_inference_tensor(input, operation, input.scalar_type());
  TORCH_CHECK(
      input.layout() == at::kStrided,
      operation,
      " supports only strided tensors");
  TORCH_CHECK(
      input.dim() <= 8,
      operation,
      " supports at most eight dimensions");
}

void write_ne_shape(
    const at::Tensor& input,
    std::uint32_t* sizes,
    std::uint32_t* strides,
    const char* operation) {
  for (const auto dim : c10::irange(input.dim())) {
    sizes[dim] = checked_u32(input.size(dim), "ne.Tensor input size");
    TORCH_CHECK(
        input.stride(dim) >= 0,
        operation,
        " does not support negative strides");
    strides[dim] = checked_u32(input.stride(dim), "ne.Tensor input stride");
  }
}

at::Tensor ne_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  constexpr const char* operation = "WebGPU ne.Tensor integer profile";
  check_ne_input(lhs, operation);
  check_ne_input(rhs, operation);
  TORCH_CHECK(
      lhs.device() == rhs.device(),
      operation,
      " requires both inputs on the same WebGPU device");

  auto output_shape = at::infer_size(lhs.sizes(), rhs.sizes());
  TORCH_CHECK(
      output_shape.size() <= 8,
      operation,
      " broadcast result has more than eight dimensions");
  auto output = at::empty(
      output_shape,
      lhs.options().dtype(at::kBool),
      c10::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }
  validate_storage_span(lhs, operation);
  validate_storage_span(rhs, operation);
  validate_storage_span(output, operation);

  NeTensorParams params{};
  params.length = checked_u32(output.numel(), "ne.Tensor output length");
  params.ndim = checked_u32(output.dim(), "ne.Tensor output rank");
  params.lhs_ndim = checked_u32(lhs.dim(), "ne.Tensor lhs rank");
  params.rhs_ndim = checked_u32(rhs.dim(), "ne.Tensor rhs rank");
  params.lhs_offset =
      checked_u32(lhs.storage_offset(), "ne.Tensor lhs offset");
  params.rhs_offset =
      checked_u32(rhs.storage_offset(), "ne.Tensor rhs offset");
  params.input_kinds =
      (lhs.scalar_type() == at::kLong ? 1u : 0u) |
      (rhs.scalar_type() == at::kLong ? 2u : 0u);
  for (const auto dim : c10::irange(output.dim())) {
    params.output_sizes[dim] =
        checked_u32(output.size(dim), "ne.Tensor output size");
  }
  write_ne_shape(
      lhs, params.lhs_sizes, params.lhs_strides, operation);
  write_ne_shape(
      rhs, params.rhs_sizes, params.rhs_strides, operation);

  const auto [dispatch_x, dispatch_y] = dispatch_words(params.length);
  params.dispatch_x = dispatch_x;
  auto params_buffer = make_params_buffer("ne.Tensor params", params);
  dispatch(
      ne_tensor_kernel(),
      {tensor_entry(0, lhs),
       tensor_entry(1, rhs),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      dispatch_x,
      dispatch_y);
  return output;
}

struct AllBoolParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t input_offset;
  std::uint32_t padding;
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
};

static_assert(sizeof(AllBoolParams) == 80);

at::Tensor all_bool(const at::Tensor& input) {
  check_inference_tensor(input, "WebGPU all input", at::kBool);
  TORCH_CHECK(
      input.layout() == at::kStrided,
      "WebGPU all supports only strided Bool tensors");
  TORCH_CHECK(input.dim() <= 8, "WebGPU all supports at most 8 dimensions");

  auto output = at::empty({}, input.options());
  if (input.numel() == 0) {
    output.fill_(true);
    return output;
  }
  validate_storage_span(input, "WebGPU all input");
  validate_storage_span(output, "WebGPU all output");

  AllBoolParams params{};
  params.length = checked_u32(input.numel(), "all element count");
  params.ndim = checked_u32(input.dim(), "all input rank");
  params.input_offset = checked_u32(input.storage_offset(), "all input offset");
  for (const auto dim : c10::irange(input.dim())) {
    params.sizes[dim] = checked_u32(input.size(dim), "all input size");
    params.input_strides[dim] =
        checked_u32(input.stride(dim), "all input stride");
  }

  auto params_buffer = make_params_buffer("all Bool params", params);
  dispatch(
      all_bool_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      1);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("eq.Scalar", TORCH_FN(eq_scalar));
  module.impl("ne.Tensor", TORCH_FN(ne_tensor));
  module.impl("all", TORCH_FN(all_bool));
}

} // namespace pyodide_pytorch::webgpu::llm
