#include "llm_common.h"

#include <ATen/ExpandUtils.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaximumDimensions = 8;
constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kMaximumWorkgroups = 65535;

ComputeKernel& masked_fill_scalar_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch masked_fill.Scalar",
      shaders::kMaskedFillScalar,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct DispatchShape {
  std::uint32_t x;
  std::uint32_t y;
};

DispatchShape dispatch_shape(
    std::uint32_t elements,
    const char* operation) {
  const auto groups =
      (static_cast<std::uint64_t>(elements) + kWorkgroupSize - 1) /
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
  TORCH_INTERNAL_ASSERT(tensor.numel() != 0);
  const auto storage_offset = tensor.storage_offset();
  TORCH_CHECK(
      storage_offset >= 0 &&
          static_cast<std::uint64_t>(storage_offset) <=
              std::numeric_limits<std::uint32_t>::max(),
      operation,
      " storage offset does not fit uint32 WebGPU metadata");
  std::uint64_t maximum_index =
      static_cast<std::uint64_t>(storage_offset);
  for (const auto dim : c10::irange(tensor.dim())) {
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        operation,
        " does not support negative strides");
    TORCH_CHECK(
        static_cast<std::uint64_t>(tensor.size(dim)) <=
            std::numeric_limits<std::uint32_t>::max(),
        operation,
        " dimension does not fit uint32 WebGPU metadata");
    TORCH_CHECK(
        static_cast<std::uint64_t>(tensor.stride(dim)) <=
            std::numeric_limits<std::uint32_t>::max(),
        operation,
        " stride does not fit uint32 WebGPU metadata");
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
      maximum_index <= std::numeric_limits<std::uint32_t>::max(),
      operation,
      " maximum storage index does not fit uint32 WebGPU metadata");
  TORCH_CHECK(
      maximum_index <
          allocation(tensor).buffer.GetSize() /
              static_cast<std::uint64_t>(tensor.element_size()),
      operation,
      " view exceeds its GPUBuffer storage");
}

void write_split(
    std::uint32_t dim,
    std::uint32_t value,
    std::uint32_t* first,
    std::uint32_t* second) {
  (dim < 4 ? first : second)[dim < 4 ? dim : dim - 4] = value;
}

void write_output_shape(
    c10::IntArrayRef shape,
    std::uint32_t* sizes0,
    std::uint32_t* sizes1) {
  for (const auto dim : c10::irange(shape.size())) {
    write_split(
        static_cast<std::uint32_t>(dim),
        checked_u32(shape[dim], "masked_fill.Scalar output size"),
        sizes0,
        sizes1);
  }
}

void write_tensor_shape(
    const at::Tensor& tensor,
    std::uint32_t* sizes0,
    std::uint32_t* sizes1,
    std::uint32_t* strides0,
    std::uint32_t* strides1) {
  for (const auto dim : c10::irange(tensor.dim())) {
    const auto metadata_dim = static_cast<std::uint32_t>(dim);
    write_split(
        metadata_dim,
        checked_u32(tensor.size(dim), "masked_fill.Scalar input size"),
        sizes0,
        sizes1);
    write_split(
        metadata_dim,
        checked_u32(tensor.stride(dim), "masked_fill.Scalar input stride"),
        strides0,
        strides1);
  }
}

std::uint32_t float_bits(float value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

float checked_fill_value(
    const at::Scalar& value,
    const char* operation) {
  TORCH_CHECK(
      value.isIntegral(true) || value.isFloatingPoint(),
      operation,
      " requires a real scalar value");
  // Scalar::toFloat performs ATen's checked conversion: finite values that do
  // not fit float32 fail, while representable finite values, infinities, and
  // NaNs retain their normal float32 semantics.
  return value.toFloat();
}

struct MaskedFillScalarParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t self_ndim;
  std::uint32_t mask_ndim;
  std::uint32_t self_offset;
  std::uint32_t mask_offset;
  std::uint32_t output_offset;
  std::uint32_t value_bits;
  std::uint32_t dispatch_x;
  std::uint32_t padding[3];
  std::uint32_t output_sizes0[4];
  std::uint32_t output_sizes1[4];
  std::uint32_t self_sizes0[4];
  std::uint32_t self_sizes1[4];
  std::uint32_t self_strides0[4];
  std::uint32_t self_strides1[4];
  std::uint32_t mask_sizes0[4];
  std::uint32_t mask_sizes1[4];
  std::uint32_t mask_strides0[4];
  std::uint32_t mask_strides1[4];
};

static_assert(sizeof(MaskedFillScalarParams) == 208);

at::Tensor masked_fill_scalar(
    const at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  constexpr const char* operation = "WebGPU masked_fill.Scalar";
  check_strided(self, operation, at::kFloat);
  check_strided(mask, operation, at::kBool);
  TORCH_CHECK(
      self.device() == mask.device(),
      operation,
      " requires self and mask on the same WebGPU device");
  const auto fill_value = checked_fill_value(value, operation);

  const auto output_shape = at::infer_size(self.sizes(), mask.sizes());
  TORCH_CHECK(
      output_shape.size() <= kMaximumDimensions,
      operation,
      " broadcast result has more than eight dimensions");
  auto output = at::empty(
      output_shape,
      self.options(),
      c10::MemoryFormat::Contiguous);
  if (output.numel() == 0) {
    return output;
  }

  validate_storage_span(self, operation);
  validate_storage_span(mask, operation);
  validate_storage_span(output, operation);
  TORCH_INTERNAL_ASSERT(output.storage_offset() == 0);

  MaskedFillScalarParams params{};
  params.length = checked_u32(
      output.numel(), "masked_fill.Scalar output length");
  params.ndim = checked_u32(
      output.dim(), "masked_fill.Scalar output rank");
  params.self_ndim = checked_u32(
      self.dim(), "masked_fill.Scalar self rank");
  params.mask_ndim = checked_u32(
      mask.dim(), "masked_fill.Scalar mask rank");
  params.self_offset = checked_u32(
      self.storage_offset(), "masked_fill.Scalar self offset");
  params.mask_offset = checked_u32(
      mask.storage_offset(), "masked_fill.Scalar mask offset");
  params.output_offset = checked_u32(
      output.storage_offset(), "masked_fill.Scalar output offset");
  params.value_bits = float_bits(fill_value);
  write_output_shape(
      output_shape, params.output_sizes0, params.output_sizes1);
  write_tensor_shape(
      self,
      params.self_sizes0,
      params.self_sizes1,
      params.self_strides0,
      params.self_strides1);
  write_tensor_shape(
      mask,
      params.mask_sizes0,
      params.mask_sizes1,
      params.mask_strides0,
      params.mask_strides1);

  const auto shape = dispatch_shape(params.length, operation);
  params.dispatch_x = shape.x;
  auto params_buffer = make_params_buffer(
      "masked_fill.Scalar params", params);
  dispatch(
      masked_fill_scalar_kernel(),
      {tensor_entry(0, self),
       tensor_entry(1, mask),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("masked_fill.Scalar", TORCH_FN(masked_fill_scalar));
}

} // namespace pyodide_pytorch::webgpu::llm
