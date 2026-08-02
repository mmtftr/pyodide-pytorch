#pragma once

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/library.h>
#include <webgpu/webgpu_cpp.h>

#include "core/webgpu_allocator.h"
#include "core/webgpu_context.h"
#include "embedded_shaders.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pyodide_pytorch::webgpu::llm {

using torch_webgpu::core::WebGPUAllocation;

inline std::uint32_t checked_u32(
    std::int64_t value,
    const char* description) {
  TORCH_CHECK(
      value >= 0 &&
          static_cast<std::uint64_t>(value) <=
              std::numeric_limits<std::uint32_t>::max(),
      description,
      " does not fit in uint32 WebGPU metadata");
  return static_cast<std::uint32_t>(value);
}

inline void check_inference_tensor(
    const at::Tensor& tensor,
    const char* operation,
    std::optional<at::ScalarType> dtype = std::nullopt) {
  TORCH_CHECK(
      tensor.device().is_privateuseone(), operation, " expects a WebGPU tensor");
  if (dtype) {
    TORCH_CHECK(
        tensor.scalar_type() == *dtype,
        operation,
        " received unsupported dtype ",
        tensor.scalar_type());
  } else {
    TORCH_CHECK(
        tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kInt ||
            tensor.scalar_type() == at::kLong || tensor.scalar_type() == at::kBool,
        operation,
        " supports only torch.float32, torch.int32, signed-int32-valued "
        "torch.int64, and byte-packed torch.bool");
  }
  TORCH_CHECK(
      !at::GradMode::is_enabled() || !tensor.requires_grad(),
      "WebGPU autograd is not supported; use `torch.no_grad()` for inference");
  TORCH_CHECK(
      tensor.storage_offset() >= 0,
      operation,
      " does not support negative storage offsets");
}

inline WebGPUAllocation& allocation(const at::Tensor& tensor) {
  check_inference_tensor(tensor, "WebGPU buffer access");
  auto* result =
      static_cast<WebGPUAllocation*>(tensor.storage().data_ptr().get());
  TORCH_CHECK(result, "WebGPU tensor has no GPUBuffer storage");
  return *result;
}

struct ComputeKernel {
  wgpu::BindGroupLayout bind_group_layout;
  wgpu::ComputePipeline pipeline;
};

inline ComputeKernel make_kernel(
    const char* label,
    const char* shader,
    const std::vector<wgpu::BufferBindingType>& binding_types) {
  const std::string shader_string(shader);
  wgpu::ShaderSourceWGSL shader_source{
      wgpu::ShaderSourceWGSL::Init{
          nullptr,
          wgpu::StringView{shader_string.c_str(), shader_string.size()}}};
  wgpu::ShaderModuleDescriptor shader_descriptor{};
  shader_descriptor.nextInChain = &shader_source;
  shader_descriptor.label = label;
  auto& context = torch_webgpu::core::getWebGPUContext();
  auto shader_module =
      context.getDevice().CreateShaderModule(&shader_descriptor);

  std::vector<wgpu::BindGroupLayoutEntry> entries(binding_types.size());
  for (std::size_t index = 0; index < binding_types.size(); ++index) {
    entries[index].binding = static_cast<std::uint32_t>(index);
    entries[index].visibility = wgpu::ShaderStage::Compute;
    entries[index].buffer.type = binding_types[index];
  }
  wgpu::BindGroupLayoutDescriptor layout_descriptor{};
  layout_descriptor.entryCount = entries.size();
  layout_descriptor.entries = entries.data();
  auto bind_group_layout =
      context.getDevice().CreateBindGroupLayout(&layout_descriptor);

  wgpu::PipelineLayoutDescriptor pipeline_layout_descriptor{};
  pipeline_layout_descriptor.bindGroupLayoutCount = 1;
  pipeline_layout_descriptor.bindGroupLayouts = &bind_group_layout;
  auto pipeline_layout =
      context.getDevice().CreatePipelineLayout(&pipeline_layout_descriptor);

  wgpu::ComputePipelineDescriptor pipeline_descriptor{};
  pipeline_descriptor.layout = pipeline_layout;
  pipeline_descriptor.compute.module = shader_module;
  pipeline_descriptor.compute.entryPoint = wgpu::StringView{"main", 4};
  auto pipeline =
      context.getDevice().CreateComputePipeline(&pipeline_descriptor);
  TORCH_CHECK(
      bind_group_layout != nullptr && pipeline != nullptr,
      "failed to create ",
      label,
      " WebGPU pipeline");
  return {std::move(bind_group_layout), std::move(pipeline)};
}

template <typename Params>
wgpu::Buffer make_params_buffer(const char* label, const Params& params) {
  auto& context = torch_webgpu::core::getWebGPUContext();
  wgpu::BufferDescriptor descriptor{};
  descriptor.label = label;
  descriptor.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  descriptor.size = sizeof(Params);
  auto buffer = context.getDevice().CreateBuffer(&descriptor);
  context.getQueue().WriteBuffer(buffer, 0, &params, sizeof(params));
  return buffer;
}

inline wgpu::BindGroupEntry tensor_entry(
    std::uint32_t binding,
    const at::Tensor& tensor) {
  auto& tensor_allocation = allocation(tensor);
  wgpu::BindGroupEntry entry{};
  entry.binding = binding;
  entry.buffer = tensor_allocation.buffer;
  entry.size = tensor_allocation.buffer.GetSize();
  return entry;
}

inline wgpu::BindGroupEntry buffer_entry(
    std::uint32_t binding,
    const wgpu::Buffer& buffer,
    std::uint64_t size) {
  wgpu::BindGroupEntry entry{};
  entry.binding = binding;
  entry.buffer = buffer;
  entry.size = size;
  return entry;
}

inline void dispatch(
    ComputeKernel& kernel,
    const std::vector<wgpu::BindGroupEntry>& entries,
    std::uint32_t x,
    std::uint32_t y = 1,
    std::uint32_t z = 1) {
  auto& context = torch_webgpu::core::getWebGPUContext();
  wgpu::BindGroupDescriptor bind_group_descriptor{};
  bind_group_descriptor.layout = kernel.bind_group_layout;
  bind_group_descriptor.entryCount = entries.size();
  bind_group_descriptor.entries = entries.data();
  auto bind_group =
      context.getDevice().CreateBindGroup(&bind_group_descriptor);
  auto encoder = context.getDevice().CreateCommandEncoder();
  auto pass = encoder.BeginComputePass();
  pass.SetPipeline(kernel.pipeline);
  pass.SetBindGroup(0, bind_group);
  pass.DispatchWorkgroups(x, y, z);
  pass.End();
  auto command_buffer = encoder.Finish();
  context.getQueue().Submit(1, &command_buffer);
}

inline std::vector<std::int64_t> concrete_sizes(
    c10::SymIntArrayRef sizes) {
  std::vector<std::int64_t> result;
  result.reserve(sizes.size());
  for (const auto& size : sizes) {
    result.push_back(size.expect_int());
  }
  return result;
}

inline std::int64_t product(c10::IntArrayRef sizes) {
  std::int64_t result = 1;
  for (const auto size : sizes) {
    TORCH_CHECK(size >= 0, "negative tensor dimension");
    TORCH_CHECK(
        size == 0 || result <= std::numeric_limits<std::int64_t>::max() / size,
        "tensor shape product overflow");
    result *= size;
  }
  return result;
}

// GPU-only copy used when an operator's output aliases one of its inputs.
// WebGPU does not allow one buffer to be bound for both read-only and writable
// storage in the same compute pass.
void copy_strided(const at::Tensor& source, at::Tensor& destination);

} // namespace pyodide_pytorch::webgpu::llm
