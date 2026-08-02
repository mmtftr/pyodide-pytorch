#include "llm_common.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaximumDimensions = 8;
constexpr std::uint32_t kMaximumTestElements = 256;
constexpr std::uint32_t kWorkgroupSize = 64;
constexpr std::uint32_t kMaximumWorkgroups = 65535;

ComputeKernel& long_cumsum_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch restricted Long cumsum",
      shaders::kLongCumsum,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& long_isin_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch restricted Long isin",
      shaders::kLongIsin,
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
    std::uint32_t units,
    const char* operation) {
  TORCH_INTERNAL_ASSERT(units > 0);
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

void check_long_tensor(
    const at::Tensor& tensor,
    const char* operation) {
  check_inference_tensor(tensor, operation, at::kLong);
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
      checked_u32(tensor.storage_offset(), "restricted Long storage offset"));
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
      "restricted Long maximum storage index");
  const auto element_bytes =
      static_cast<std::uint64_t>(tensor.element_size());
  TORCH_CHECK(
      maximum_index < allocation(tensor).buffer.GetSize() / element_bytes,
      operation,
      " view exceeds its GPUBuffer storage");
  if (tensor.scalar_type() == at::kLong) {
    TORCH_CHECK(
        maximum_index <= std::numeric_limits<std::uint32_t>::max() / 2u,
        operation,
        " Long word address exceeds WGSL uint32 indexing");
  }
}

struct LongCumsumParams {
  std::uint32_t rows;
  std::uint32_t columns;
  std::uint32_t input_offset;
  std::uint32_t output_offset;
  std::uint32_t dispatch_x;
  std::uint32_t padding[3];
};

static_assert(sizeof(LongCumsumParams) == 32);

at::Tensor long_cumsum(
    const at::Tensor& input,
    std::int64_t dim,
    std::optional<at::ScalarType> dtype) {
  constexpr const char* operation = "WebGPU restricted Long cumsum";
  check_long_tensor(input, operation);
  TORCH_CHECK(
      !dtype || *dtype == at::kLong,
      operation,
      " supports only torch.int64 output");
  TORCH_CHECK(
      input.dim() > 0,
      operation,
      " requires a tensor with at least one dimension");
  const auto normalized_dim = dim < 0 ? dim + input.dim() : dim;
  TORCH_CHECK(
      normalized_dim >= 0 && normalized_dim < input.dim(),
      operation,
      " dimension out of range");
  TORCH_CHECK(
      normalized_dim == input.dim() - 1,
      operation,
      " supports only the last dimension");
  TORCH_CHECK(
      input.is_contiguous(),
      operation,
      " requires contiguous input");

  auto output = at::empty(
      input.sizes(), input.options(), c10::MemoryFormat::Contiguous);
  if (input.numel() == 0) {
    return output;
  }
  TORCH_INTERNAL_ASSERT(input.size(-1) > 0);
  validate_storage_span(input, operation);
  validate_storage_span(output, operation);

  LongCumsumParams params{};
  params.columns = checked_u32(input.size(-1), "Long cumsum column count");
  params.rows = checked_u32(
      input.numel() / input.size(-1), "Long cumsum row count");
  params.input_offset =
      checked_u32(input.storage_offset(), "Long cumsum input offset");
  params.output_offset =
      checked_u32(output.storage_offset(), "Long cumsum output offset");
  const auto shape = dispatch_shape(params.rows, operation);
  params.dispatch_x = shape.x;

  auto params_buffer = make_params_buffer("Long cumsum params", params);
  dispatch(
      long_cumsum_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

struct LongIsinParams {
  std::uint32_t length;
  std::uint32_t elements_ndim;
  std::uint32_t elements_offset;
  std::uint32_t test_length;
  std::uint32_t test_offset;
  std::uint32_t test_stride;
  std::uint32_t invert;
  std::uint32_t dispatch_x;
  std::uint32_t elements_sizes[8];
  std::uint32_t elements_strides[8];
};

static_assert(sizeof(LongIsinParams) == 96);

at::Tensor long_isin_tensor_tensor(
    const at::Tensor& elements,
    const at::Tensor& test_elements,
    bool assume_unique,
    bool invert) {
  constexpr const char* operation =
      "WebGPU restricted Long isin.Tensor_Tensor";
  check_long_tensor(elements, operation);
  check_long_tensor(test_elements, operation);
  TORCH_CHECK(
      elements.device() == test_elements.device(),
      operation,
      " requires both tensors on the same WebGPU device");
  TORCH_CHECK(
      test_elements.dim() <= 1,
      operation,
      " supports only scalar or one-dimensional test_elements");
  TORCH_CHECK(
      test_elements.numel() <= kMaximumTestElements,
      operation,
      " supports at most 256 test elements");
  // Direct membership testing has identical results for unique and duplicate
  // test sets, so both values of the optimization hint are safe.
  (void)assume_unique;

  auto output = at::empty(
      elements.sizes(),
      elements.options().dtype(at::kBool),
      c10::MemoryFormat::Contiguous);
  if (elements.numel() == 0) {
    return output;
  }
  validate_storage_span(elements, operation);
  validate_storage_span(output, operation);
  if (test_elements.numel() != 0) {
    validate_storage_span(test_elements, operation);
  }

  LongIsinParams params{};
  params.length = checked_u32(elements.numel(), "Long isin element count");
  params.elements_ndim =
      checked_u32(elements.dim(), "Long isin elements rank");
  params.elements_offset =
      checked_u32(elements.storage_offset(), "Long isin elements offset");
  params.test_length =
      checked_u32(test_elements.numel(), "Long isin test-element count");
  params.test_offset = checked_u32(
      test_elements.storage_offset(), "Long isin test-elements offset");
  if (test_elements.dim() == 1) {
    TORCH_CHECK(
        test_elements.stride(0) >= 0,
        operation,
        " does not support negative test-element strides");
    params.test_stride = checked_u32(
        test_elements.stride(0), "Long isin test-elements stride");
  }
  params.invert = invert ? 1u : 0u;
  for (const auto dim_index : c10::irange(elements.dim())) {
    TORCH_CHECK(
        elements.stride(dim_index) >= 0,
        operation,
        " does not support negative element strides");
    params.elements_sizes[dim_index] =
        checked_u32(elements.size(dim_index), "Long isin elements size");
    params.elements_strides[dim_index] = checked_u32(
        elements.stride(dim_index), "Long isin elements stride");
  }

  const auto output_words = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(params.length) + 3u) / 4u);
  const auto shape = dispatch_shape(output_words, operation);
  params.dispatch_x = shape.x;

  // Empty tensors may not own a bindable GPUBuffer. The shader performs no
  // test-set reads when test_length is zero, so one inaccessible dummy Long is
  // sufficient for the required binding.
  at::Tensor empty_test_dummy;
  const at::Tensor* test_binding = &test_elements;
  if (test_elements.numel() == 0) {
    empty_test_dummy = at::empty({1}, test_elements.options());
    test_binding = &empty_test_dummy;
  }

  auto params_buffer = make_params_buffer("Long isin params", params);
  dispatch(
      long_isin_kernel(),
      {tensor_entry(0, elements),
       tensor_entry(1, *test_binding),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("cumsum", TORCH_FN(long_cumsum));
  module.impl(
      "isin.Tensor_Tensor", TORCH_FN(long_isin_tensor_tensor));
}

} // namespace pyodide_pytorch::webgpu::llm
