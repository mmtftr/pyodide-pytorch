#include "llm_common.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& gemma_rms_norm_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch Gemma offset-weight RMS norm",
      shaders::kGemmaRmsNorm,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

void check_contiguous_float_storage(
    const at::Tensor& tensor,
    const char* operation) {
  check_inference_tensor(tensor, operation, at::kFloat);
  TORCH_CHECK(
      tensor.layout() == at::kStrided && tensor.is_contiguous(),
      operation,
      " requires contiguous strided tensors");
  if (tensor.numel() == 0) {
    return;
  }
  const auto offset = checked_u32(
      tensor.storage_offset(), "Gemma RMSNorm storage offset");
  const auto elements = checked_u32(
      tensor.numel(), "Gemma RMSNorm element count");
  TORCH_CHECK(
      elements - 1 <= std::numeric_limits<std::uint32_t>::max() - offset,
      operation,
      " storage span does not fit uint32 WebGPU metadata");
  TORCH_CHECK(
      static_cast<std::uint64_t>(offset) + elements <=
          allocation(tensor).buffer.GetSize() / sizeof(float),
      operation,
      " tensor exceeds its GPUBuffer storage");
}

std::pair<std::uint32_t, std::uint32_t> dispatch_rows(std::uint32_t rows) {
  constexpr std::uint32_t maximum_workgroups = 65535;
  const auto x = std::min(rows, maximum_workgroups);
  const auto y = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(rows) + x - 1) / x);
  TORCH_CHECK(
      y <= maximum_workgroups,
      "WebGPU Gemma RMSNorm dispatch exceeds WebGPU limits");
  return {x, y};
}

struct GemmaRmsNormParams {
  std::uint32_t rows;
  std::uint32_t width;
  std::uint32_t input_offset;
  std::uint32_t output_offset;
  std::uint32_t weight_offset;
  std::uint32_t dispatch_x;
  float epsilon;
  std::uint32_t padding;
};

static_assert(sizeof(GemmaRmsNormParams) == 32);

at::Tensor gemma_rms_norm(
    const at::Tensor& input,
    const at::Tensor& offset_weight,
    double epsilon) {
  constexpr const char* operation = "WebGPU gemma_rms_norm";
  check_inference_tensor(input, operation, at::kFloat);
  check_contiguous_float_storage(offset_weight, operation);
  TORCH_CHECK(
      input.layout() == at::kStrided,
      operation,
      " requires strided input");
  TORCH_CHECK(
      input.device() == offset_weight.device(),
      operation,
      " requires input and weight on the same WebGPU device");
  TORCH_CHECK(
      input.dim() >= 1 && input.dim() <= 8,
      operation,
      " expects input ranks one through eight");
  TORCH_CHECK(
      offset_weight.dim() == 1 &&
          offset_weight.size(0) == input.size(-1),
      operation,
      " offset-weight shape must match the input's last dimension");
  TORCH_CHECK(
      input.size(-1) > 0,
      operation,
      " does not support an empty normalized dimension");
  TORCH_CHECK(
      std::isfinite(epsilon) && epsilon >= 0.0 &&
          epsilon <= std::numeric_limits<float>::max(),
      operation,
      " requires a finite nonnegative epsilon");

  auto contiguous_input = input.is_contiguous() ? input : input.contiguous();
  auto output = at::empty(
      input.sizes(), input.options(), c10::MemoryFormat::Contiguous);
  if (input.numel() == 0) {
    return output;
  }
  check_contiguous_float_storage(contiguous_input, operation);
  check_contiguous_float_storage(output, operation);

  const auto width = input.size(-1);
  const auto rows = input.numel() / width;
  const auto [dispatch_x, dispatch_y] = dispatch_rows(
      checked_u32(rows, "Gemma RMSNorm row count"));
  GemmaRmsNormParams params{};
  params.rows = checked_u32(rows, "Gemma RMSNorm row count");
  params.width = checked_u32(width, "Gemma RMSNorm width");
  params.input_offset = checked_u32(
      contiguous_input.storage_offset(), "Gemma RMSNorm input offset");
  params.output_offset = checked_u32(
      output.storage_offset(), "Gemma RMSNorm output offset");
  params.weight_offset = checked_u32(
      offset_weight.storage_offset(), "Gemma RMSNorm weight offset");
  params.dispatch_x = dispatch_x;
  params.epsilon = static_cast<float>(epsilon);

  auto params_buffer = make_params_buffer("Gemma RMSNorm params", params);
  dispatch(
      gemma_rms_norm_kernel(),
      {tensor_entry(0, contiguous_input),
       tensor_entry(1, offset_weight),
       tensor_entry(2, output),
       buffer_entry(3, params_buffer, sizeof(params))},
      dispatch_x,
      dispatch_y);
  return output;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(webgpu, module) {
  module.def(
      "gemma_rms_norm(Tensor input, Tensor offset_weight, float epsilon) "
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, module) {
  module.impl("gemma_rms_norm", TORCH_FN(gemma_rms_norm));
}

} // namespace pyodide_pytorch::webgpu::llm
