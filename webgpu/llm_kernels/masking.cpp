#include "llm_common.h"

#include <ATen/ExpandUtils.h>
#include <ATen/ops/mul_ops.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaximumDimensions = 8;
constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kMaximumWorkgroups = 65535;

ComputeKernel& gt_tensor_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch gt tensor",
      shaders::kGtTensor,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& triangular_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch triangular",
      shaders::kTriangular,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& where_float_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch where float",
      shaders::kWhereFloat,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& mul_bool_inplace_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch float by Bool inplace multiply",
      shaders::kMulBoolInplace,
      {wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct DispatchShape {
  std::uint32_t x;
  std::uint32_t y;
};

DispatchShape dispatch_shape(
    std::uint32_t units,
    const char* operation) {
  const auto groups =
      (static_cast<std::uint64_t>(units) + kWorkgroupSize - 1) /
      kWorkgroupSize;
  const auto x = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(groups, kMaximumWorkgroups));
  const auto y = static_cast<std::uint32_t>((groups + x - 1) / x);
  TORCH_CHECK(
      y <= kMaximumWorkgroups,
      operation,
      " dispatch exceeds WebGPU limits");
  return {x, y};
}

void check_strided(
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
      checked_u32(tensor.storage_offset(), "mask storage offset"));
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
  checked_u32(
      static_cast<std::int64_t>(maximum_index),
      "mask maximum storage index");
  TORCH_CHECK(
      maximum_index <
          allocation(tensor).buffer.GetSize() /
              static_cast<std::uint64_t>(tensor.element_size()),
      operation,
      " view exceeds its GPUBuffer storage");
}

void check_same_device(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    const char* operation) {
  TORCH_CHECK(
      lhs.device() == rhs.device(),
      operation,
      " requires tensors on the same WebGPU device");
}

template <typename Params>
void write_shape(
    Params& params,
    const at::Tensor& tensor,
    std::uint32_t* sizes,
    std::uint32_t* strides,
    const char* operation) {
  for (const auto dim : c10::irange(tensor.dim())) {
    sizes[dim] = checked_u32(tensor.size(dim), "mask tensor size");
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        operation,
        " does not support negative strides");
    strides[dim] = checked_u32(tensor.stride(dim), "mask tensor stride");
  }
}

void write_output_shape(
    c10::IntArrayRef shape,
    std::uint32_t* sizes) {
  for (const auto dim : c10::irange(shape.size())) {
    sizes[dim] = checked_u32(shape[dim], "mask output size");
  }
}

std::uint32_t float_bits(float value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct GtTensorParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t lhs_ndim;
  std::uint32_t rhs_ndim;
  std::uint32_t lhs_offset;
  std::uint32_t rhs_offset;
  std::uint32_t input_kind;
  std::uint32_t dispatch_x;
  std::uint32_t output_sizes[8];
  std::uint32_t lhs_sizes[8];
  std::uint32_t lhs_strides[8];
  std::uint32_t rhs_sizes[8];
  std::uint32_t rhs_strides[8];
};

static_assert(sizeof(GtTensorParams) == 192);

at::Tensor gt_tensor(
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  constexpr const char* operation = "WebGPU gt.Tensor";
  TORCH_CHECK(
      lhs.scalar_type() == at::kFloat || lhs.scalar_type() == at::kInt ||
          lhs.scalar_type() == at::kLong,
      operation,
      " supports only torch.float32, torch.int32, and signed-int32-valued "
      "torch.int64");
  TORCH_CHECK(
      rhs.scalar_type() == lhs.scalar_type(),
      operation,
      " requires matching input dtypes");
  check_strided(lhs, operation, lhs.scalar_type());
  check_strided(rhs, operation, rhs.scalar_type());
  check_same_device(lhs, rhs, operation);

  auto output_shape = at::infer_size(lhs.sizes(), rhs.sizes());
  TORCH_CHECK(
      output_shape.size() <= kMaximumDimensions,
      operation,
      " broadcast result has more than eight dimensions");
  auto output = at::empty(
      output_shape,
      lhs.options().dtype(at::kBool),
      c10::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }
  TORCH_INTERNAL_ASSERT(output.storage_offset() == 0);
  validate_storage_span(lhs, operation);
  validate_storage_span(rhs, operation);
  validate_storage_span(output, operation);

  GtTensorParams params{};
  params.length = checked_u32(output.numel(), "gt.Tensor output length");
  params.ndim = checked_u32(output.dim(), "gt.Tensor output rank");
  params.lhs_ndim = checked_u32(lhs.dim(), "gt.Tensor lhs rank");
  params.rhs_ndim = checked_u32(rhs.dim(), "gt.Tensor rhs rank");
  params.lhs_offset = checked_u32(lhs.storage_offset(), "gt.Tensor lhs offset");
  params.rhs_offset = checked_u32(rhs.storage_offset(), "gt.Tensor rhs offset");
  params.input_kind = lhs.scalar_type() == at::kFloat
      ? 0
      : (lhs.scalar_type() == at::kInt ? 1 : 2);
  write_output_shape(output_shape, params.output_sizes);
  write_shape(params, lhs, params.lhs_sizes, params.lhs_strides, operation);
  write_shape(params, rhs, params.rhs_sizes, params.rhs_strides, operation);

  const auto words = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(params.length) + 3) / 4);
  const auto shape = dispatch_shape(words, operation);
  params.dispatch_x = shape.x;
  auto params_buffer = make_params_buffer("gt.Tensor params", params);
  dispatch(
      gt_tensor_kernel(),
      {tensor_entry(0, lhs),
       tensor_entry(1, rhs),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

enum class TriangularOperation : std::uint32_t {
  Upper = 0,
  Lower = 1,
};

struct TriangularParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t input_offset;
  std::uint32_t output_offset;
  std::uint32_t operation;
  std::uint32_t output_kind;
  std::uint32_t dispatch_x;
  std::int32_t diagonal;
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
};

static_assert(sizeof(TriangularParams) == 96);

at::Tensor triangular(
    const at::Tensor& input,
    std::int64_t diagonal,
    TriangularOperation triangular_operation,
    const char* operation) {
  TORCH_CHECK(
      input.scalar_type() == at::kFloat || input.scalar_type() == at::kBool,
      operation,
      " supports only torch.float32 and torch.bool");
  check_strided(input, operation, input.scalar_type());
  TORCH_CHECK(input.dim() >= 2, operation, ": input tensor must have at least 2 dimensions");
  TORCH_CHECK(
      diagonal >= std::numeric_limits<std::int32_t>::min() &&
          diagonal <= std::numeric_limits<std::int32_t>::max(),
      operation,
      " diagonal is outside the signed-int32 WebGPU profile");
  TORCH_CHECK(
      input.size(-1) <= std::numeric_limits<std::int32_t>::max() &&
          input.size(-2) <= std::numeric_limits<std::int32_t>::max(),
      operation,
      " matrix dimensions exceed the signed-int32 WebGPU profile");

  auto output = at::empty(
      input.sizes(), input.options(), c10::MemoryFormat::Contiguous);
  if (input.numel() == 0) {
    return output;
  }
  validate_storage_span(input, operation);
  validate_storage_span(output, operation);
  if (input.scalar_type() == at::kBool) {
    TORCH_INTERNAL_ASSERT(output.storage_offset() == 0);
  }

  TriangularParams params{};
  params.length = checked_u32(input.numel(), "triangular output length");
  params.ndim = checked_u32(input.dim(), "triangular input rank");
  params.input_offset = checked_u32(input.storage_offset(), "triangular input offset");
  params.output_offset = checked_u32(output.storage_offset(), "triangular output offset");
  params.operation = static_cast<std::uint32_t>(triangular_operation);
  params.output_kind = input.scalar_type() == at::kFloat ? 0 : 1;
  params.diagonal = static_cast<std::int32_t>(diagonal);
  for (const auto dim : c10::irange(input.dim())) {
    params.sizes[dim] = checked_u32(input.size(dim), "triangular input size");
    TORCH_CHECK(
        input.stride(dim) >= 0,
        operation,
        " does not support negative strides");
    params.input_strides[dim] =
        checked_u32(input.stride(dim), "triangular input stride");
  }
  const auto units = params.output_kind == 0
      ? params.length
      : static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(params.length) + 3) / 4);
  const auto shape = dispatch_shape(units, operation);
  params.dispatch_x = shape.x;
  auto params_buffer = make_params_buffer("triangular params", params);
  dispatch(
      triangular_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

at::Tensor triu(const at::Tensor& input, std::int64_t diagonal) {
  return triangular(
      input, diagonal, TriangularOperation::Upper, "WebGPU triu");
}

at::Tensor tril(const at::Tensor& input, std::int64_t diagonal) {
  return triangular(
      input, diagonal, TriangularOperation::Lower, "WebGPU tril");
}

struct WhereParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t condition_ndim;
  std::uint32_t lhs_ndim;
  std::uint32_t rhs_ndim;
  std::uint32_t condition_offset;
  std::uint32_t lhs_offset;
  std::uint32_t rhs_offset;
  std::uint32_t lhs_is_scalar;
  std::uint32_t rhs_is_scalar;
  std::uint32_t lhs_scalar_bits;
  std::uint32_t rhs_scalar_bits;
  std::uint32_t output_offset;
  std::uint32_t dispatch_x;
  std::uint32_t padding[2];
  std::uint32_t output_sizes[8];
  std::uint32_t condition_sizes[8];
  std::uint32_t condition_strides[8];
  std::uint32_t lhs_sizes[8];
  std::uint32_t lhs_strides[8];
  std::uint32_t rhs_sizes[8];
  std::uint32_t rhs_strides[8];
};

static_assert(sizeof(WhereParams) == 288);

bool is_cpu_float_scalar(const at::Tensor& tensor) {
  return tensor.device().is_cpu() && tensor.layout() == at::kStrided &&
      tensor.scalar_type() == at::kFloat && tensor.dim() == 0 &&
      tensor.numel() == 1 && !tensor.requires_grad();
}

void check_where_value(
    const at::Tensor& tensor,
    const at::Tensor& condition,
    const char* operation) {
  TORCH_CHECK(
      tensor.scalar_type() == at::kFloat,
      operation,
      " supports only float32 result tensors");
  if (is_cpu_float_scalar(tensor)) {
    return;
  }
  check_strided(tensor, operation, at::kFloat);
  check_same_device(tensor, condition, operation);
  validate_storage_span(tensor, operation);
}

at::Tensor where_self(
    const at::Tensor& condition,
    const at::Tensor& lhs,
    const at::Tensor& rhs) {
  constexpr const char* operation = "WebGPU where.self";
  check_strided(condition, operation, at::kBool);
  check_where_value(lhs, condition, operation);
  check_where_value(rhs, condition, operation);
  auto output_shape = at::infer_size(condition.sizes(), lhs.sizes());
  output_shape = at::infer_size(output_shape, rhs.sizes());
  TORCH_CHECK(
      output_shape.size() <= kMaximumDimensions,
      operation,
      " broadcast result has more than eight dimensions");
  auto output = at::empty(
      output_shape,
      condition.options().dtype(at::kFloat),
      c10::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }
  validate_storage_span(condition, operation);
  validate_storage_span(output, operation);

  const auto lhs_scalar = is_cpu_float_scalar(lhs);
  const auto rhs_scalar = is_cpu_float_scalar(rhs);
  WhereParams params{};
  params.length = checked_u32(output.numel(), "where.self output length");
  params.ndim = checked_u32(output.dim(), "where.self output rank");
  params.condition_ndim = checked_u32(condition.dim(), "where.self condition rank");
  params.lhs_ndim = lhs_scalar ? 0 : checked_u32(lhs.dim(), "where.self lhs rank");
  params.rhs_ndim = rhs_scalar ? 0 : checked_u32(rhs.dim(), "where.self rhs rank");
  params.condition_offset =
      checked_u32(condition.storage_offset(), "where.self condition offset");
  params.lhs_offset = lhs_scalar
      ? 0
      : checked_u32(lhs.storage_offset(), "where.self lhs offset");
  params.rhs_offset = rhs_scalar
      ? 0
      : checked_u32(rhs.storage_offset(), "where.self rhs offset");
  params.lhs_is_scalar = lhs_scalar ? 1 : 0;
  params.rhs_is_scalar = rhs_scalar ? 1 : 0;
  params.lhs_scalar_bits = lhs_scalar ? float_bits(lhs.item<float>()) : 0;
  params.rhs_scalar_bits = rhs_scalar ? float_bits(rhs.item<float>()) : 0;
  params.output_offset = checked_u32(output.storage_offset(), "where.self output offset");
  write_output_shape(output_shape, params.output_sizes);
  write_shape(
      params,
      condition,
      params.condition_sizes,
      params.condition_strides,
      operation);
  if (!lhs_scalar) {
    write_shape(params, lhs, params.lhs_sizes, params.lhs_strides, operation);
  }
  if (!rhs_scalar) {
    write_shape(params, rhs, params.rhs_sizes, params.rhs_strides, operation);
  }

  const auto shape = dispatch_shape(params.length, operation);
  params.dispatch_x = shape.x;
  const auto& lhs_binding = lhs_scalar ? condition : lhs;
  const auto& rhs_binding = rhs_scalar ? condition : rhs;
  auto params_buffer = make_params_buffer("where.self params", params);
  dispatch(
      where_float_kernel(),
      {tensor_entry(0, condition),
       tensor_entry(1, lhs_binding),
       tensor_entry(2, rhs_binding),
       tensor_entry(3, output),
       buffer_entry(4, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

struct MulBoolInplaceParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t rhs_ndim;
  std::uint32_t lhs_offset;
  std::uint32_t rhs_offset;
  std::uint32_t dispatch_x;
  std::uint32_t padding[2];
  std::uint32_t sizes[8];
  std::uint32_t lhs_strides[8];
  std::uint32_t rhs_sizes[8];
  std::uint32_t rhs_strides[8];
};

static_assert(sizeof(MulBoolInplaceParams) == 160);

at::Tensor& mul_bool_inplace(
    at::Tensor& lhs,
    const at::Tensor& rhs) {
  constexpr const char* operation = "WebGPU mul_.Tensor Float-by-Bool";
  check_strided(lhs, operation, at::kFloat);
  check_strided(rhs, operation, at::kBool);
  check_same_device(lhs, rhs, operation);
  TORCH_CHECK(
      lhs.is_contiguous(),
      operation,
      " requires a contiguous left-hand tensor");
  auto broadcast_shape = at::infer_size(lhs.sizes(), rhs.sizes());
  TORCH_CHECK(
      c10::IntArrayRef(broadcast_shape) == lhs.sizes(),
      operation,
      " cannot expand the left-hand tensor in-place");
  TORCH_CHECK(
      allocation(lhs).buffer.Get() != allocation(rhs).buffer.Get(),
      operation,
      " does not support aliased Float and Bool storage");
  if (lhs.numel() == 0) {
    return lhs;
  }
  validate_storage_span(lhs, operation);
  validate_storage_span(rhs, operation);

  MulBoolInplaceParams params{};
  params.length = checked_u32(lhs.numel(), "mul_.Tensor output length");
  params.ndim = checked_u32(lhs.dim(), "mul_.Tensor lhs rank");
  params.rhs_ndim = checked_u32(rhs.dim(), "mul_.Tensor rhs rank");
  params.lhs_offset = checked_u32(lhs.storage_offset(), "mul_.Tensor lhs offset");
  params.rhs_offset = checked_u32(rhs.storage_offset(), "mul_.Tensor rhs offset");
  for (const auto dim : c10::irange(lhs.dim())) {
    params.sizes[dim] = checked_u32(lhs.size(dim), "mul_.Tensor lhs size");
    params.lhs_strides[dim] = checked_u32(lhs.stride(dim), "mul_.Tensor lhs stride");
  }
  write_shape(params, rhs, params.rhs_sizes, params.rhs_strides, operation);
  const auto shape = dispatch_shape(params.length, operation);
  params.dispatch_x = shape.x;
  auto params_buffer = make_params_buffer("mul_.Tensor Float-by-Bool params", params);
  dispatch(
      mul_bool_inplace_kernel(),
      {tensor_entry(0, lhs),
       tensor_entry(1, rhs),
       buffer_entry(2, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return lhs;
}

at::Tensor& mul_tensor_inplace(
    at::Tensor& lhs,
    const at::Tensor& rhs) {
  if (lhs.scalar_type() == at::kFloat && rhs.scalar_type() == at::kBool) {
    return mul_bool_inplace(lhs, rhs);
  }

  // Registering a dtype-specialized kernel still replaces mul_.Tensor for the
  // entire PrivateUse1 dispatch key. Preserve the backend's normal in-place
  // behavior by routing every non-mask dtype pair through its existing
  // mul.out kernel. That implementation already handles exact output/input
  // aliasing with a temporary GPU buffer.
  return at::_ops::mul_out::call(lhs, rhs, lhs);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("gt.Tensor", TORCH_FN(gt_tensor));
  module.impl("triu", TORCH_FN(triu));
  module.impl("tril", TORCH_FN(tril));
  module.impl("where.self", TORCH_FN(where_self));
  module.impl("mul_.Tensor", TORCH_FN(mul_tensor_inplace));
}

} // namespace pyodide_pytorch::webgpu::llm
