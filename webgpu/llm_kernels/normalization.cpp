#include "llm_common.h"

#include <algorithm>
#include <tuple>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& layer_norm_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch layer norm",
      shaders::kLayerNorm,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& rms_norm_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch RMS norm",
      shaders::kRmsNorm,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

std::vector<std::int64_t> validate_normalized_shape(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const char* operation) {
  auto shape = concrete_sizes(normalized_shape);
  TORCH_CHECK(!shape.empty() && shape.size() <= static_cast<std::size_t>(input.dim()),
      operation, " received an invalid normalized_shape");
  const auto start = input.dim() - static_cast<std::int64_t>(shape.size());
  for (const auto index : c10::irange(shape.size())) {
    TORCH_CHECK(input.size(start + index) == shape[index],
        operation, " normalized_shape does not match the input suffix");
  }
  TORCH_CHECK(product(shape) > 0, operation, " does not support empty normalized dimensions");
  return shape;
}

std::pair<std::uint32_t, std::uint32_t> dispatch_rows(std::uint32_t rows) {
  constexpr std::uint32_t max_workgroups = 65535;
  const auto x = std::min(rows, max_workgroups);
  const auto y = (rows + x - 1) / x;
  TORCH_CHECK(y <= max_workgroups, "WebGPU normalization dispatch is too large");
  return {x, y};
}

struct LayerNormParams {
  std::uint32_t rows;
  std::uint32_t width;
  std::uint32_t input_offset;
  std::uint32_t output_offset;
  std::uint32_t mean_offset;
  std::uint32_t rstd_offset;
  std::uint32_t weight_offset;
  std::uint32_t bias_offset;
  std::uint32_t has_weight;
  std::uint32_t has_bias;
  std::uint32_t dispatch_x;
  std::uint32_t padding0;
  float epsilon;
  std::uint32_t padding[3];
};

static_assert(sizeof(LayerNormParams) == 64);

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_impl(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    double epsilon) {
  check_inference_tensor(input, "WebGPU layer_norm input", at::kFloat);
  auto shape = validate_normalized_shape(input, normalized_shape, "WebGPU layer_norm");
  const auto width = product(shape);
  const auto rows = input.numel() / width;
  auto contiguous_input = input.is_contiguous() ? input : input.contiguous();
  auto output = at::empty(input.sizes(), input.options());

  std::vector<std::int64_t> statistic_shape;
  const auto prefix_dimensions = input.dim() - static_cast<std::int64_t>(shape.size());
  for (const auto dim : c10::irange(prefix_dimensions)) {
    statistic_shape.push_back(input.size(dim));
  }
  statistic_shape.insert(statistic_shape.end(), shape.size(), 1);
  auto mean = at::empty(statistic_shape, input.options());
  auto rstd = at::empty(statistic_shape, input.options());
  if (input.numel() == 0) {
    return {output, mean, rstd};
  }

  at::Tensor contiguous_weight;
  at::Tensor contiguous_bias;
  if (weight && weight->defined()) {
    check_inference_tensor(*weight, "WebGPU layer_norm weight", at::kFloat);
    TORCH_CHECK(weight->sizes() == at::IntArrayRef(shape),
        "WebGPU layer_norm weight shape mismatch");
    contiguous_weight = weight->is_contiguous() ? *weight : weight->contiguous();
  }
  if (bias && bias->defined()) {
    check_inference_tensor(*bias, "WebGPU layer_norm bias", at::kFloat);
    TORCH_CHECK(bias->sizes() == at::IntArrayRef(shape),
        "WebGPU layer_norm bias shape mismatch");
    contiguous_bias = bias->is_contiguous() ? *bias : bias->contiguous();
  }
  const auto [dispatch_x, dispatch_y] =
      dispatch_rows(checked_u32(rows, "layer_norm row count"));

  LayerNormParams params{};
  params.rows = checked_u32(rows, "layer_norm row count");
  params.width = checked_u32(width, "layer_norm width");
  params.input_offset = checked_u32(
      contiguous_input.storage_offset(), "layer_norm input offset");
  params.output_offset =
      checked_u32(output.storage_offset(), "layer_norm output offset");
  params.mean_offset =
      checked_u32(mean.storage_offset(), "layer_norm mean offset");
  params.rstd_offset =
      checked_u32(rstd.storage_offset(), "layer_norm rstd offset");
  params.has_weight = contiguous_weight.defined() ? 1 : 0;
  params.has_bias = contiguous_bias.defined() ? 1 : 0;
  params.weight_offset = contiguous_weight.defined()
      ? checked_u32(contiguous_weight.storage_offset(), "layer_norm weight offset")
      : 0;
  params.bias_offset = contiguous_bias.defined()
      ? checked_u32(contiguous_bias.storage_offset(), "layer_norm bias offset")
      : 0;
  params.dispatch_x = dispatch_x;
  params.epsilon = static_cast<float>(epsilon);

  const auto& weight_binding =
      contiguous_weight.defined() ? contiguous_weight : contiguous_input;
  const auto& bias_binding =
      contiguous_bias.defined() ? contiguous_bias : contiguous_input;
  auto params_buffer = make_params_buffer("layer_norm params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, contiguous_input),
      tensor_entry(1, weight_binding),
      tensor_entry(2, bias_binding),
      tensor_entry(3, output),
      tensor_entry(4, mean),
      tensor_entry(5, rstd),
      buffer_entry(6, params_buffer, sizeof(params))};
  dispatch(layer_norm_kernel(), entries, dispatch_x, dispatch_y, 1);
  return {output, mean, rstd};
}

struct RmsNormParams {
  std::uint32_t rows;
  std::uint32_t width;
  std::uint32_t input_offset;
  std::uint32_t output_offset;
  std::uint32_t weight_offset;
  std::uint32_t has_weight;
  std::uint32_t dispatch_x;
  std::uint32_t padding0;
  float epsilon;
  std::uint32_t padding[3];
};

static_assert(sizeof(RmsNormParams) == 48);

at::Tensor rms_norm_impl(
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight,
    std::optional<double> epsilon) {
  check_inference_tensor(input, "WebGPU rms_norm input", at::kFloat);
  auto shape = validate_normalized_shape(input, normalized_shape, "WebGPU rms_norm");
  const auto width = product(shape);
  const auto rows = input.numel() / width;
  auto contiguous_input = input.is_contiguous() ? input : input.contiguous();
  auto output = at::empty(input.sizes(), input.options());
  if (input.numel() == 0) {
    return output;
  }

  at::Tensor contiguous_weight;
  if (weight && weight->defined()) {
    check_inference_tensor(*weight, "WebGPU rms_norm weight", at::kFloat);
    TORCH_CHECK(weight->sizes() == at::IntArrayRef(shape),
        "WebGPU rms_norm weight shape mismatch");
    contiguous_weight = weight->is_contiguous() ? *weight : weight->contiguous();
  }
  const auto [dispatch_x, dispatch_y] =
      dispatch_rows(checked_u32(rows, "rms_norm row count"));
  RmsNormParams params{};
  params.rows = checked_u32(rows, "rms_norm row count");
  params.width = checked_u32(width, "rms_norm width");
  params.input_offset = checked_u32(
      contiguous_input.storage_offset(), "rms_norm input offset");
  params.output_offset =
      checked_u32(output.storage_offset(), "rms_norm output offset");
  params.has_weight = contiguous_weight.defined() ? 1 : 0;
  params.weight_offset = contiguous_weight.defined()
      ? checked_u32(contiguous_weight.storage_offset(), "rms_norm weight offset")
      : 0;
  params.dispatch_x = dispatch_x;
  params.epsilon = static_cast<float>(
      epsilon.value_or(std::numeric_limits<float>::epsilon()));

  const auto& weight_binding =
      contiguous_weight.defined() ? contiguous_weight : contiguous_input;
  auto params_buffer = make_params_buffer("rms_norm params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, contiguous_input),
      tensor_entry(1, weight_binding),
      tensor_entry(2, output),
      buffer_entry(3, params_buffer, sizeof(params))};
  dispatch(rms_norm_kernel(), entries, dispatch_x, dispatch_y, 1);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("native_layer_norm", TORCH_FN(native_layer_norm_impl));
  module.impl("rms_norm", TORCH_FN(rms_norm_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
