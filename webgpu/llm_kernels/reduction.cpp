#include "llm_common.h"

#include <algorithm>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& mean_dim_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch dimension mean",
      shaders::kMeanDim,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

std::pair<std::uint32_t, std::uint32_t> dispatch_rows(
    std::uint32_t rows) {
  constexpr std::uint32_t max_workgroups = 65535;
  const auto x = std::min(rows, max_workgroups);
  const auto y = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(rows) + x - 1) / x);
  TORCH_CHECK(y <= max_workgroups, "WebGPU mean dispatch is too large");
  return {x, y};
}

void validate_float_storage_span(
    const at::Tensor& tensor,
    const char* operation) {
  if (tensor.numel() == 0) {
    return;
  }
  std::uint64_t maximum_index = checked_u32(
      tensor.storage_offset(), "mean storage offset");
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
      maximum_index < allocation(tensor).nbytes / sizeof(float),
      operation,
      " view exceeds its GPUBuffer storage");
  checked_u32(
      static_cast<std::int64_t>(maximum_index), "mean maximum storage index");
}

struct MeanDimParams {
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

static_assert(sizeof(MeanDimParams) == 96);

at::Tensor mean_dim_impl(
    const at::Tensor& input,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {
  check_inference_tensor(input, "WebGPU mean.dim input", at::kFloat);
  TORCH_CHECK(
      !dtype || *dtype == at::kFloat,
      "WebGPU mean.dim supports only torch.float32 output");
  TORCH_CHECK(
      dim && dim->size() == 1,
      "WebGPU mean.dim requires exactly one reduction dimension");
  TORCH_CHECK(
      input.dim() <= 8,
      "WebGPU mean.dim supports at most 8 dimensions");

  const auto rank = input.dim();
  const auto requested_dim = dim->front();
  std::int64_t normalized_dim = 0;
  if (rank == 0) {
    TORCH_CHECK(
        requested_dim == 0 || requested_dim == -1,
        "WebGPU mean.dim dimension out of range for a scalar");
  } else {
    normalized_dim = requested_dim < 0 ? requested_dim + rank : requested_dim;
    TORCH_CHECK(
        normalized_dim >= 0 && normalized_dim < rank,
        "WebGPU mean.dim dimension out of range");
  }

  auto output_sizes = input.sizes().vec();
  if (rank != 0) {
    if (keepdim) {
      output_sizes[normalized_dim] = 1;
    } else {
      output_sizes.erase(output_sizes.begin() + normalized_dim);
    }
  }
  auto output = at::empty(output_sizes, input.options());
  if (output.numel() == 0) {
    return output;
  }

  const auto reduce_size = rank == 0 ? 1 : input.size(normalized_dim);
  const auto reduce_stride = rank == 0 ? 1 : input.stride(normalized_dim);
  TORCH_CHECK(
      reduce_stride >= 0,
      "WebGPU mean.dim does not support negative strides");
  if (reduce_size != 0) {
    validate_float_storage_span(input, "WebGPU mean.dim");
  }
  validate_float_storage_span(output, "WebGPU mean.dim output");

  const auto rows = checked_u32(output.numel(), "mean output element count");
  const auto [dispatch_x, dispatch_y] = dispatch_rows(rows);
  MeanDimParams params{};
  params.output_length = rows;
  params.reduce_size = checked_u32(reduce_size, "mean reduction size");
  params.ndim = checked_u32(rank, "mean input rank");
  params.reduce_dim = checked_u32(normalized_dim, "mean reduction dimension");
  params.input_offset =
      checked_u32(input.storage_offset(), "mean input offset");
  params.output_offset =
      checked_u32(output.storage_offset(), "mean output offset");
  params.reduce_stride = checked_u32(reduce_stride, "mean reduction stride");
  params.dispatch_x = dispatch_x;
  for (const auto current_dim : c10::irange(rank)) {
    params.sizes[current_dim] =
        checked_u32(input.size(current_dim), "mean input size");
    params.input_strides[current_dim] =
        checked_u32(input.stride(current_dim), "mean input stride");
  }

  // An empty reduction still has nonempty output, but its input owns no
  // GPUBuffer. Bind one harmless float word; the shader performs no reads.
  at::Tensor empty_reduction_dummy;
  const at::Tensor* input_binding = &input;
  if (reduce_size == 0) {
    empty_reduction_dummy = at::empty({1}, input.options());
    input_binding = &empty_reduction_dummy;
  }

  auto params_buffer = make_params_buffer("mean.dim params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, *input_binding),
      tensor_entry(1, output),
      buffer_entry(2, params_buffer, sizeof(params))};
  dispatch(mean_dim_kernel(), entries, dispatch_x, dispatch_y, 1);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("mean.dim", TORCH_FN(mean_dim_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
