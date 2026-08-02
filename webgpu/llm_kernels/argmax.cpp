#include "llm_common.h"

#include <algorithm>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& argmax_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch last-dimension argmax",
      shaders::kArgmax,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

std::pair<std::uint32_t, std::uint32_t> dispatch_rows(
    std::uint32_t rows) {
  constexpr std::uint32_t max_workgroups = 65535;
  const auto x = std::min(rows, max_workgroups);
  const auto y = (rows + x - 1) / x;
  TORCH_CHECK(y <= max_workgroups, "WebGPU argmax dispatch is too large");
  return {x, y};
}

void validate_float_storage_span(const at::Tensor& tensor) {
  if (tensor.numel() == 0) {
    return;
  }
  std::uint64_t maximum_index = checked_u32(
      tensor.storage_offset(), "argmax storage offset");
  for (const auto dim : c10::irange(tensor.dim())) {
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        "WebGPU argmax does not support negative strides");
    const auto size = static_cast<std::uint64_t>(tensor.size(dim));
    const auto stride = static_cast<std::uint64_t>(tensor.stride(dim));
    if (size > 0) {
      TORCH_CHECK(
          stride == 0 ||
              size - 1 <=
                  (std::numeric_limits<std::uint64_t>::max() - maximum_index) /
                      stride,
          "WebGPU argmax storage span overflow");
      maximum_index += (size - 1) * stride;
    }
  }
  TORCH_CHECK(
      maximum_index < allocation(tensor).buffer.GetSize() / sizeof(float),
      "WebGPU argmax view exceeds its GPUBuffer storage");
  checked_u32(
      static_cast<std::int64_t>(maximum_index),
      "argmax maximum storage index");
}

struct ArgmaxParams {
  std::uint32_t output_length;
  std::uint32_t reduce_size;
  std::uint32_t ndim;
  std::uint32_t reduce_dim;
  std::uint32_t input_offset;
  std::uint32_t output_offset;
  std::uint32_t reduce_stride;
  std::uint32_t dispatch_x;
  std::uint32_t sizes[8];
  std::uint32_t input_strides[8];
};

static_assert(sizeof(ArgmaxParams) == 96);

at::Tensor argmax_impl(
    const at::Tensor& input,
    std::optional<std::int64_t> dim,
    bool keepdim) {
  check_inference_tensor(input, "WebGPU argmax input", at::kFloat);
  TORCH_CHECK(
      input.dim() > 0 && input.dim() <= 8,
      "WebGPU argmax supports input ranks 1 through 8");
  TORCH_CHECK(
      dim.has_value(),
      "WebGPU argmax requires an explicit last-dimension reduction");

  const auto rank = input.dim();
  const auto normalized_dim = *dim < 0 ? *dim + rank : *dim;
  TORCH_CHECK(
      normalized_dim >= 0 && normalized_dim < rank,
      "WebGPU argmax dimension out of range");
  TORCH_CHECK(
      normalized_dim == rank - 1,
      "WebGPU argmax supports only the last dimension");

  const auto reduce_size = input.size(normalized_dim);
  TORCH_CHECK(
      reduce_size > 0,
      "argmax(): Expected reduction dim ",
      normalized_dim,
      " to have non-zero size.");
  TORCH_CHECK(
      reduce_size <= std::numeric_limits<std::int32_t>::max(),
      "WebGPU argmax indices exceed the signed-int32 Long profile");

  auto output_sizes = input.sizes().vec();
  if (keepdim) {
    output_sizes[normalized_dim] = 1;
  } else {
    output_sizes.pop_back();
  }
  auto output = at::empty(output_sizes, input.options().dtype(at::kLong));
  if (output.numel() == 0) {
    return output;
  }

  validate_float_storage_span(input);
  const auto rows = checked_u32(
      output.numel(), "argmax output element count");
  const auto [dispatch_x, dispatch_y] = dispatch_rows(rows);

  ArgmaxParams params{};
  params.output_length = rows;
  params.reduce_size = checked_u32(reduce_size, "argmax reduction size");
  params.ndim = checked_u32(rank, "argmax input rank");
  params.reduce_dim = checked_u32(
      normalized_dim, "argmax reduction dimension");
  params.input_offset = checked_u32(
      input.storage_offset(), "argmax input offset");
  params.output_offset = checked_u32(
      output.storage_offset(), "argmax output offset");
  params.reduce_stride = checked_u32(
      input.stride(normalized_dim), "argmax reduction stride");
  params.dispatch_x = dispatch_x;
  for (const auto current_dim : c10::irange(rank)) {
    params.sizes[current_dim] = checked_u32(
        input.size(current_dim), "argmax input size");
    params.input_strides[current_dim] = checked_u32(
        input.stride(current_dim), "argmax input stride");
  }

  auto params_buffer = make_params_buffer("argmax params", params);
  dispatch(
      argmax_kernel(),
      {tensor_entry(0, input),
       tensor_entry(1, output),
       buffer_entry(2, params_buffer, sizeof(params))},
      dispatch_x,
      dispatch_y);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("argmax", TORCH_FN(argmax_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
