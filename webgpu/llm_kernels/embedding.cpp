#include "llm_common.h"

#include <algorithm>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& embedding_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch embedding",
      shaders::kEmbedding,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct EmbeddingParams {
  std::uint32_t num_indices;
  std::uint32_t embedding_dim;
  std::uint32_t num_embeddings;
  std::uint32_t weight_offset;
  std::uint32_t indices_offset;
  std::uint32_t output_offset;
  std::uint32_t weight_stride0;
  std::uint32_t weight_stride1;
  std::uint32_t dispatch_x;
  std::uint32_t index_words;
  std::uint32_t padding[2];
};

static_assert(sizeof(EmbeddingParams) == 48);

at::Tensor embedding_impl(
    const at::Tensor& weight,
    const at::Tensor& indices,
    c10::SymInt /*padding_idx*/,
    bool scale_grad_by_freq,
    bool sparse) {
  check_inference_tensor(weight, "WebGPU embedding weight", at::kFloat);
  check_inference_tensor(indices, "WebGPU embedding indices");
  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "WebGPU embedding indices must be torch.int32 or signed-int32-valued "
      "torch.int64");
  TORCH_CHECK(weight.dim() == 2, "WebGPU embedding weight must be 2-D");
  TORCH_CHECK(!scale_grad_by_freq && !sparse,
      "WebGPU embedding supports inference-only forward semantics");
  auto contiguous_indices =
      indices.is_contiguous() ? indices : indices.contiguous();

  auto output_shape = indices.sizes().vec();
  output_shape.push_back(weight.size(1));
  auto output = at::empty(output_shape, weight.options());
  if (output.numel() == 0) {
    return output;
  }

  const auto workgroups =
      (checked_u32(output.numel(), "embedding output length") + 63) / 64;
  constexpr std::uint32_t max_workgroups = 65535;
  const auto dispatch_x = std::min(workgroups, max_workgroups);
  const auto dispatch_y = (workgroups + dispatch_x - 1) / dispatch_x;
  TORCH_CHECK(dispatch_y <= max_workgroups, "WebGPU embedding dispatch is too large");

  EmbeddingParams params{};
  params.num_indices = checked_u32(indices.numel(), "embedding index count");
  params.embedding_dim = checked_u32(weight.size(1), "embedding dimension");
  params.num_embeddings = checked_u32(weight.size(0), "embedding row count");
  params.weight_offset =
      checked_u32(weight.storage_offset(), "embedding weight offset");
  params.indices_offset = checked_u32(
      contiguous_indices.storage_offset(), "embedding index offset");
  params.output_offset =
      checked_u32(output.storage_offset(), "embedding output offset");
  params.weight_stride0 =
      checked_u32(weight.stride(0), "embedding weight stride 0");
  params.weight_stride1 =
      checked_u32(weight.stride(1), "embedding weight stride 1");
  params.dispatch_x = dispatch_x;
  params.index_words = indices.scalar_type() == at::kLong ? 2 : 1;

  auto params_buffer = make_params_buffer("embedding params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, weight),
      tensor_entry(1, contiguous_indices),
      tensor_entry(2, output),
      buffer_entry(3, params_buffer, sizeof(params))};
  dispatch(embedding_kernel(), entries, dispatch_x, dispatch_y, 1);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("embedding", TORCH_FN(embedding_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
