#include "llm_common.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaxWorkgroupsPerDimension = 65535;
constexpr std::uint32_t kPortableColumns = 64;
constexpr std::uint32_t kSubgroupColumns = 4;

#if defined(__EMSCRIPTEN__)
EM_JS(int, swiglu_subgroup_s4_supported, (), {
  const state = globalThis.__torchWebGPU;
  if (!state || !state.device || state.fusedSwiGluVariant === "portable") {
    return 0;
  }
  const device = state.device;
  const info = device.adapterInfo;
  return device.features && device.features.has("subgroups") && info &&
      info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
});
#else
int swiglu_subgroup_s4_supported() {
  return 0;
}
#endif

ComputeKernel& swiglu_gemv_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch fused SwiGLU GEMV",
      shaders::kSwiGluGemv,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& swiglu_gemv_subgroup_s4_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch fused SwiGLU GEMV subgroup s4",
      shaders::kSwiGluGemvSubgroupS4,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

void check_contiguous_float(
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
      tensor.storage_offset(), "fused SwiGLU storage offset");
  const auto elements = checked_u32(
      tensor.numel(), "fused SwiGLU tensor element count");
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

void check_optional_bias(
    const std::optional<at::Tensor>& bias,
    const at::Tensor& input,
    std::int64_t columns,
    const char* operation) {
  if (!bias || !bias->defined()) {
    return;
  }
  check_contiguous_float(*bias, operation);
  TORCH_CHECK(
      bias->device() == input.device(),
      operation,
      " requires every tensor on the same WebGPU device");
  TORCH_CHECK(
      bias->dim() == 1 && bias->size(0) == columns,
      operation,
      " bias shape mismatch");
}

struct SwiGluParams {
  std::uint32_t columns;
  std::uint32_t inner;
  std::uint32_t input_offset;
  std::uint32_t gate_weight_offset;
  std::uint32_t up_weight_offset;
  std::uint32_t gate_bias_offset;
  std::uint32_t up_bias_offset;
  std::uint32_t output_offset;
  std::uint32_t has_gate_bias;
  std::uint32_t has_up_bias;
  std::uint32_t padding[6];
};

static_assert(sizeof(SwiGluParams) == 64);

at::Tensor fused_swiglu_impl(
    const at::Tensor& input,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    const std::optional<at::Tensor>& gate_bias,
    const std::optional<at::Tensor>& up_bias) {
  constexpr const char* operation = "WebGPU fused_swiglu";
  check_contiguous_float(input, operation);
  check_contiguous_float(gate_weight, operation);
  check_contiguous_float(up_weight, operation);
  TORCH_CHECK(
      input.dim() >= 1 && input.dim() <= 8,
      operation,
      " expects input ranks 1 through 8");
  TORCH_CHECK(
      gate_weight.dim() == 2 && up_weight.dim() == 2,
      operation,
      " expects 2-D gate and up weights");
  TORCH_CHECK(
      gate_weight.device() == input.device() &&
          up_weight.device() == input.device(),
      operation,
      " requires every tensor on the same WebGPU device");
  TORCH_CHECK(
      gate_weight.sizes() == up_weight.sizes(),
      operation,
      " requires matching gate and up weight shapes");
  TORCH_CHECK(
      input.size(-1) == gate_weight.size(1),
      operation,
      " input/weight feature mismatch");

  const auto rows = product(input.sizes().slice(0, input.dim() - 1));
  const auto inner = input.size(-1);
  const auto columns = gate_weight.size(0);
  TORCH_CHECK(
      rows == 1,
      operation,
      " supports exactly one flattened decode row");
  TORCH_CHECK(
      inner > 0 && columns > 0,
      operation,
      " requires nonempty input and intermediate dimensions");
  check_optional_bias(gate_bias, input, columns, operation);
  check_optional_bias(up_bias, input, columns, operation);

  auto output_shape = input.sizes().vec();
  output_shape.back() = columns;
  auto output = at::empty(
      output_shape, input.options(), c10::MemoryFormat::Contiguous);
  check_contiguous_float(output, operation);

  SwiGluParams params{};
  params.columns = checked_u32(columns, "fused SwiGLU intermediate size");
  params.inner = checked_u32(inner, "fused SwiGLU hidden size");
  params.input_offset =
      checked_u32(input.storage_offset(), "fused SwiGLU input offset");
  params.gate_weight_offset = checked_u32(
      gate_weight.storage_offset(), "fused SwiGLU gate weight offset");
  params.up_weight_offset = checked_u32(
      up_weight.storage_offset(), "fused SwiGLU up weight offset");
  params.gate_bias_offset = gate_bias && gate_bias->defined()
      ? checked_u32(
            gate_bias->storage_offset(), "fused SwiGLU gate bias offset")
      : 0;
  params.up_bias_offset = up_bias && up_bias->defined()
      ? checked_u32(up_bias->storage_offset(), "fused SwiGLU up bias offset")
      : 0;
  params.output_offset =
      checked_u32(output.storage_offset(), "fused SwiGLU output offset");
  params.has_gate_bias = gate_bias && gate_bias->defined() ? 1 : 0;
  params.has_up_bias = up_bias && up_bias->defined() ? 1 : 0;

  const auto& gate_bias_binding = params.has_gate_bias ? *gate_bias : input;
  const auto& up_bias_binding = params.has_up_bias ? *up_bias : input;
  auto params_buffer = make_params_buffer("fused SwiGLU params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, input),
      tensor_entry(1, gate_weight),
      tensor_entry(2, up_weight),
      tensor_entry(3, gate_bias_binding),
      tensor_entry(4, up_bias_binding),
      tensor_entry(5, output),
      buffer_entry(6, params_buffer, sizeof(params))};

  if (swiglu_subgroup_s4_supported()) {
    const auto workgroups =
        (params.columns + kSubgroupColumns - 1) / kSubgroupColumns;
    if (workgroups <= kMaxWorkgroupsPerDimension) {
      dispatch(
          swiglu_gemv_subgroup_s4_kernel(), entries, workgroups, 1, 1);
      return output;
    }
  }
  const auto workgroups =
      (params.columns + kPortableColumns - 1) / kPortableColumns;
  TORCH_CHECK(
      workgroups <= kMaxWorkgroupsPerDimension,
      operation,
      " dispatch exceeds WebGPU limits");
  dispatch(swiglu_gemv_kernel(), entries, workgroups, 1, 1);
  return output;
}

} // namespace

TORCH_LIBRARY_FRAGMENT(webgpu, module) {
  module.def(
      "fused_swiglu(Tensor input, Tensor gate_weight, Tensor up_weight, "
      "Tensor? gate_bias=None, Tensor? up_bias=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, module) {
  module.impl("fused_swiglu", TORCH_FN(fused_swiglu_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
