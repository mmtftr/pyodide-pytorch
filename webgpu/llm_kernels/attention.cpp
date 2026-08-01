#include "llm_common.h"

#include <array>
#include <cmath>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& sdpa_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch fused SDPA",
      shaders::kSdpa,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct SdpaParams {
  std::uint32_t batch;
  std::uint32_t query_heads;
  std::uint32_t key_value_heads;
  std::uint32_t query_length;
  std::uint32_t key_value_length;
  std::uint32_t head_dim;
  std::uint32_t query_offset;
  std::uint32_t key_offset;
  std::uint32_t value_offset;
  std::uint32_t output_offset;
  std::uint32_t mask_offset;
  std::uint32_t has_mask;
  std::uint32_t causal;
  std::uint32_t padding0[3];
  float scale;
  std::uint32_t padding1[3];
  std::uint32_t query_strides[4];
  std::uint32_t key_strides[4];
  std::uint32_t value_strides[4];
  std::uint32_t output_strides[4];
  std::uint32_t mask_strides[4];
};

static_assert(sizeof(SdpaParams) == 160);

void set_strides(
    std::uint32_t destination[4],
    const at::Tensor& tensor,
    const char* description) {
  TORCH_CHECK(tensor.dim() == 4, description, " expects a 4-D tensor");
  for (const auto dim : c10::irange(4)) {
    TORCH_CHECK(tensor.stride(dim) >= 0, description, " rejects negative strides");
    destination[dim] = checked_u32(tensor.stride(dim), description);
  }
}

at::Tensor sdpa_impl(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attention_mask,
    double dropout_probability,
    bool causal,
    std::optional<double> scale,
    bool enable_gqa) {
  check_inference_tensor(query, "WebGPU SDPA query", at::kFloat);
  check_inference_tensor(key, "WebGPU SDPA key", at::kFloat);
  check_inference_tensor(value, "WebGPU SDPA value", at::kFloat);
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
      "WebGPU SDPA expects [batch, heads, sequence, head_dim]");
  TORCH_CHECK(dropout_probability == 0.0,
      "WebGPU SDPA supports inference with dropout_p=0 only");
  TORCH_CHECK(query.size(0) == key.size(0) && key.size(0) == value.size(0),
      "WebGPU SDPA batch mismatch");
  TORCH_CHECK(key.size(1) == value.size(1),
      "WebGPU SDPA key/value head mismatch");
  TORCH_CHECK(key.size(2) == value.size(2),
      "WebGPU SDPA key/value sequence mismatch");
  TORCH_CHECK(query.size(3) == key.size(3) && key.size(3) == value.size(3),
      "WebGPU SDPA head dimension mismatch");
  TORCH_CHECK(query.size(1) % key.size(1) == 0,
      "WebGPU SDPA query heads must be divisible by key/value heads");
  TORCH_CHECK(query.size(1) == key.size(1) || enable_gqa,
      "WebGPU SDPA requires enable_gqa=True when head counts differ");
  TORCH_CHECK(!causal || key.size(2) >= query.size(2),
      "WebGPU causal SDPA requires key length >= query length");
  TORCH_CHECK(query.size(2) > 0 && key.size(2) > 0 && query.size(3) > 0,
      "WebGPU SDPA does not support empty attention dimensions");

  auto output = at::empty(query.sizes(), query.options());
  SdpaParams params{};
  params.batch = checked_u32(query.size(0), "SDPA batch");
  params.query_heads = checked_u32(query.size(1), "SDPA query heads");
  params.key_value_heads = checked_u32(key.size(1), "SDPA key/value heads");
  params.query_length = checked_u32(query.size(2), "SDPA query length");
  params.key_value_length = checked_u32(key.size(2), "SDPA key/value length");
  params.head_dim = checked_u32(query.size(3), "SDPA head dimension");
  params.query_offset = checked_u32(query.storage_offset(), "SDPA query offset");
  params.key_offset = checked_u32(key.storage_offset(), "SDPA key offset");
  params.value_offset = checked_u32(value.storage_offset(), "SDPA value offset");
  params.output_offset = checked_u32(output.storage_offset(), "SDPA output offset");
  params.causal = causal ? 1 : 0;
  params.scale = static_cast<float>(
      scale.value_or(1.0 / std::sqrt(static_cast<double>(query.size(3)))));
  set_strides(params.query_strides, query, "SDPA query strides");
  set_strides(params.key_strides, key, "SDPA key strides");
  set_strides(params.value_strides, value, "SDPA value strides");
  set_strides(params.output_strides, output, "SDPA output strides");

  at::Tensor mask_binding = query;
  if (attention_mask && attention_mask->defined()) {
    const auto& mask = *attention_mask;
    check_inference_tensor(mask, "WebGPU SDPA mask", at::kFloat);
    TORCH_CHECK(mask.dim() <= 4, "WebGPU SDPA mask rank must be <= 4");
    const std::array<std::int64_t, 4> target_sizes{
        query.size(0), query.size(1), query.size(2), key.size(2)};
    const auto leading = 4 - mask.dim();
    for (const auto axis : c10::irange(4)) {
      if (axis < leading) {
        params.mask_strides[axis] = 0;
        continue;
      }
      const auto mask_axis = axis - leading;
      TORCH_CHECK(
          mask.size(mask_axis) == 1 || mask.size(mask_axis) == target_sizes[axis],
          "WebGPU SDPA mask is not broadcastable to attention scores");
      params.mask_strides[axis] = mask.size(mask_axis) == 1
          ? 0
          : checked_u32(mask.stride(mask_axis), "SDPA mask stride");
    }
    params.has_mask = 1;
    params.mask_offset = checked_u32(mask.storage_offset(), "SDPA mask offset");
    mask_binding = mask;
  }
  TORCH_CHECK(params.query_length <= 65535 && params.query_heads <= 65535 &&
          params.batch <= 65535,
      "WebGPU SDPA dispatch dimensions exceed WebGPU limits");

  auto params_buffer = make_params_buffer("SDPA params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, query),
      tensor_entry(1, key),
      tensor_entry(2, value),
      tensor_entry(3, mask_binding),
      tensor_entry(4, output),
      buffer_entry(5, params_buffer, sizeof(params))};
  dispatch(
      sdpa_kernel(),
      entries,
      params.query_length,
      params.query_heads,
      params.batch);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("scaled_dot_product_attention", TORCH_FN(sdpa_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
