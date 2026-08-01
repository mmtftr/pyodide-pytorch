#include "llm_common.h"

#include <ATen/core/TensorBody.h>

#include <array>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& strided_copy_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch strided copy",
      shaders::kStridedCopy,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct CopyParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t source_offset;
  std::uint32_t destination_offset;
  std::uint32_t sizes[8];
  std::uint32_t source_strides[8];
  std::uint32_t destination_strides[8];
};

static_assert(sizeof(CopyParams) == 112);

void copy_strided(const at::Tensor& source, at::Tensor& destination) {
  check_inference_tensor(source, "WebGPU strided copy");
  check_inference_tensor(destination, "WebGPU strided copy");
  TORCH_CHECK(source.scalar_type() == destination.scalar_type(),
      "WebGPU strided copy requires matching dtypes");
  TORCH_CHECK(source.sizes() == destination.sizes(),
      "WebGPU strided copy requires matching shapes");
  TORCH_CHECK(source.dim() <= 8, "WebGPU strided copy supports at most 8 dimensions");
  if (source.numel() == 0) {
    return;
  }
  TORCH_CHECK(
      allocation(source).buffer.Get() != allocation(destination).buffer.Get(),
      "WebGPU strided copy does not support overlapping views of one buffer");

  CopyParams params{};
  params.length = checked_u32(source.numel(), "strided-copy element count");
  params.ndim = checked_u32(source.dim(), "strided-copy rank");
  params.source_offset =
      checked_u32(source.storage_offset(), "strided-copy source offset");
  params.destination_offset = checked_u32(
      destination.storage_offset(), "strided-copy destination offset");
  for (const auto dim : c10::irange(source.dim())) {
    TORCH_CHECK(source.stride(dim) >= 0 && destination.stride(dim) >= 0,
        "WebGPU strided copy does not support negative strides");
    params.sizes[dim] = checked_u32(source.size(dim), "strided-copy size");
    params.source_strides[dim] =
        checked_u32(source.stride(dim), "strided-copy source stride");
    params.destination_strides[dim] = checked_u32(
        destination.stride(dim), "strided-copy destination stride");
  }

  auto params_buffer = make_params_buffer("strided-copy params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, source),
      tensor_entry(1, destination),
      buffer_entry(2, params_buffer, sizeof(params))};
  dispatch(strided_copy_kernel(), entries, (params.length + 63) / 64);
}

at::Tensor clone_impl(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format) {
  check_inference_tensor(self, "WebGPU clone");
  const auto format = memory_format.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(
      format == c10::MemoryFormat::Contiguous ||
          format == c10::MemoryFormat::Preserve,
      "WebGPU clone supports only contiguous or preserve memory format");
  auto output = at::empty(
      self.sizes(), self.options(), c10::MemoryFormat::Contiguous);
  copy_strided(self, output);
  return output;
}

at::Tensor contiguous_impl(
    const at::Tensor& self,
    c10::MemoryFormat memory_format) {
  check_inference_tensor(self, "WebGPU contiguous");
  TORCH_CHECK(
      memory_format == c10::MemoryFormat::Contiguous,
      "WebGPU contiguous supports only contiguous_format");
  if (self.is_contiguous()) {
    return self;
  }
  return clone_impl(self, c10::MemoryFormat::Contiguous);
}

at::Tensor cat_impl(const at::ITensorListRef& tensors, std::int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "cat expects a non-empty tensor list");
  const auto& first = tensors.front();
  check_inference_tensor(first, "WebGPU cat");
  const auto rank = first.dim();
  if (dim < 0) {
    dim += rank;
  }
  TORCH_CHECK(dim >= 0 && dim < rank, "WebGPU cat dimension out of range");

  std::int64_t concatenated_size = 0;
  for (const auto& tensor : tensors) {
    check_inference_tensor(tensor, "WebGPU cat");
    TORCH_CHECK(tensor.scalar_type() == first.scalar_type(),
        "WebGPU cat requires matching dtypes");
    TORCH_CHECK(tensor.dim() == rank, "WebGPU cat requires matching ranks");
    for (const auto current_dim : c10::irange(rank)) {
      if (current_dim != dim) {
        TORCH_CHECK(tensor.size(current_dim) == first.size(current_dim),
            "WebGPU cat shape mismatch");
      }
    }
    concatenated_size += tensor.size(dim);
  }

  auto output_sizes = first.sizes().vec();
  output_sizes[dim] = concatenated_size;
  auto output = at::empty(output_sizes, first.options());
  std::int64_t offset = 0;
  for (const auto& tensor : tensors) {
    if (tensor.numel() != 0) {
      auto output_slice = output.narrow(dim, offset, tensor.size(dim));
      copy_strided(tensor, output_slice);
    }
    offset += tensor.size(dim);
  }
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("clone", TORCH_FN(clone_impl));
  module.impl("contiguous", TORCH_FN(contiguous_impl));
  module.impl("cat", TORCH_FN(cat_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
