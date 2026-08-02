#include "llm_common.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kGroupSize = 128;
constexpr std::uint32_t kValuesPerWord = 4;
constexpr std::uint32_t kOutputsPerSubgroup = 4;
constexpr std::uint32_t kMaxWorkgroupsPerDimension = 65535;

#if defined(__EMSCRIPTEN__)
EM_JS(int, q8_linear_subgroup_s4_supported, (), {
  const state = globalThis.__torchWebGPU;
  if (!state || !state.device) {
    return 0;
  }
  const device = state.device;
  const info = device.adapterInfo;
  return device.features && device.features.has("subgroups") && info &&
      info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
});
#else
int q8_linear_subgroup_s4_supported() {
  return 0;
}
#endif

ComputeKernel& q8_linear_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch Q8 linear subgroup s4",
      shaders::kLinearGemvQ8S4,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

void check_contiguous_tensor(
    const at::Tensor& tensor,
    at::ScalarType dtype,
    const char* operation) {
  check_inference_tensor(tensor, operation, dtype);
  TORCH_CHECK(
      tensor.layout() == at::kStrided && tensor.is_contiguous(),
      operation,
      " requires contiguous strided tensors");
  TORCH_CHECK(
      tensor.element_size() == sizeof(std::uint32_t),
      operation,
      " requires four-byte tensor elements");

  const auto offset =
      checked_u32(tensor.storage_offset(), "Q8 linear storage offset");
  const auto elements =
      checked_u32(tensor.numel(), "Q8 linear tensor element count");
  if (elements == 0) {
    return;
  }
  TORCH_CHECK(
      elements - 1 <= std::numeric_limits<std::uint32_t>::max() - offset,
      operation,
      " storage span does not fit uint32 WebGPU metadata");
  TORCH_CHECK(
      static_cast<std::uint64_t>(offset) + elements <=
          allocation(tensor).buffer.GetSize() / sizeof(std::uint32_t),
      operation,
      " tensor exceeds its GPUBuffer storage");
}

struct Q8LinearParams {
  std::uint32_t columns;
  std::uint32_t inner;
  std::uint32_t input_offset;
  std::uint32_t packed_weight_offset;
  std::uint32_t scale_offset;
  std::uint32_t bias_offset;
  std::uint32_t output_offset;
  std::uint32_t has_bias;
  std::uint32_t words_per_row;
  std::uint32_t groups_per_row;
  std::uint32_t group_size;
  std::uint32_t format_version;
  std::uint32_t padding[4];
};

static_assert(sizeof(Q8LinearParams) == 64);

at::Tensor q8_linear_impl(
    const at::Tensor& input,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& bias,
    std::int64_t group_size,
    std::int64_t format_version) {
  constexpr const char* operation = "WebGPU q8_linear";
  check_contiguous_tensor(input, at::kFloat, operation);
  check_contiguous_tensor(packed_weight, at::kInt, operation);
  check_contiguous_tensor(scales, at::kFloat, operation);
  TORCH_CHECK(
      input.device() == packed_weight.device() &&
          input.device() == scales.device(),
      operation,
      " requires every tensor on the same WebGPU device");
  TORCH_CHECK(
      input.dim() >= 1 && input.dim() <= 8,
      operation,
      " expects input ranks 1 through 8");
  TORCH_CHECK(
      packed_weight.dim() == 2 && scales.dim() == 2,
      operation,
      " expects packed_weight and scales to be rank-2 tensors");
  TORCH_CHECK(
      group_size == kGroupSize,
      operation,
      " supports only group_size=128");
  TORCH_CHECK(
      format_version == kFormatVersion,
      operation,
      " supports only format_version=1");

  const auto rows = product(input.sizes().slice(0, input.dim() - 1));
  const auto inner = input.size(-1);
  const auto columns = packed_weight.size(0);
  TORCH_CHECK(
      rows == 1,
      operation,
      " is decode-only and supports exactly one flattened input row");
  TORCH_CHECK(
      inner > 0 && columns > 0 && inner % kGroupSize == 0,
      operation,
      " requires nonempty dimensions and K divisible by 128");
  TORCH_CHECK(
      packed_weight.size(1) == inner / kValuesPerWord,
      operation,
      " packed_weight must have shape [N, K/4]");
  TORCH_CHECK(
      scales.size(0) == columns && scales.size(1) == inner / kGroupSize,
      operation,
      " scales must have shape [N, K/128]");

  if (bias && bias->defined()) {
    check_contiguous_tensor(*bias, at::kFloat, operation);
    TORCH_CHECK(
        bias->device() == input.device(),
        operation,
        " requires every tensor on the same WebGPU device");
    TORCH_CHECK(
        bias->dim() == 1 && bias->size(0) == columns,
        operation,
        " bias must have shape [N]");
  }
  TORCH_CHECK(
      q8_linear_subgroup_s4_supported(),
      operation,
      " requires an enabled fixed-size-32 WebGPU subgroup device");

  Q8LinearParams params{};
  params.columns = checked_u32(columns, "Q8 linear output features");
  params.inner = checked_u32(inner, "Q8 linear input features");
  params.words_per_row = checked_u32(
      packed_weight.size(1), "Q8 linear packed words per row");
  params.groups_per_row =
      checked_u32(scales.size(1), "Q8 linear groups per row");
  params.group_size = kGroupSize;
  params.format_version = kFormatVersion;
  const auto workgroups = params.columns / kOutputsPerSubgroup +
      static_cast<std::uint32_t>(
          params.columns % kOutputsPerSubgroup != 0);
  TORCH_CHECK(
      workgroups <= kMaxWorkgroupsPerDimension,
      operation,
      " dispatch exceeds WebGPU limits");

  auto output_shape = input.sizes().vec();
  output_shape.back() = columns;
  auto output = at::empty(
      output_shape, input.options(), c10::MemoryFormat::Contiguous);
  check_contiguous_tensor(output, at::kFloat, operation);

  params.input_offset =
      checked_u32(input.storage_offset(), "Q8 linear input offset");
  params.packed_weight_offset = checked_u32(
      packed_weight.storage_offset(), "Q8 linear packed-weight offset");
  params.scale_offset =
      checked_u32(scales.storage_offset(), "Q8 linear scale offset");
  params.bias_offset = bias && bias->defined()
      ? checked_u32(bias->storage_offset(), "Q8 linear bias offset")
      : 0;
  params.output_offset =
      checked_u32(output.storage_offset(), "Q8 linear output offset");
  params.has_bias = bias && bias->defined() ? 1 : 0;

  const auto& bias_binding = params.has_bias ? *bias : input;
  auto params_buffer = make_params_buffer("Q8 linear params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, input),
      tensor_entry(1, packed_weight),
      tensor_entry(2, scales),
      tensor_entry(3, bias_binding),
      tensor_entry(4, output),
      buffer_entry(5, params_buffer, sizeof(params))};
  dispatch(q8_linear_kernel(), entries, workgroups, 1, 1);
  return output;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(webgpu, module) {
  module.def(
      "q8_linear(Tensor input, Tensor packed_weight, Tensor scales, "
      "Tensor? bias=None, int group_size=128, int format_version=1) -> Tensor");
}

TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, module) {
  module.impl("q8_linear", TORCH_FN(q8_linear_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
