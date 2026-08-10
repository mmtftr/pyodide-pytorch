#include "llm_common.h"

#include <algorithm>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& int_to_float_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch int to float",
      shaders::kIntToFloat,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& bool_to_long_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch bool to restricted Long",
      shaders::kBoolToLong,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

WebGPUAllocation& cast_allocation(const at::Tensor& tensor) {
  auto* result =
      static_cast<WebGPUAllocation*>(tensor.storage().data_ptr().get());
  TORCH_CHECK(result, "WebGPU cast tensor has no GPUBuffer storage");
  return *result;
}

wgpu::BindGroupEntry cast_tensor_entry(
    std::uint32_t binding,
    const at::Tensor& tensor) {
  auto& tensor_allocation = cast_allocation(tensor);
  wgpu::BindGroupEntry entry{};
  entry.binding = binding;
  entry.buffer = tensor_allocation.buffer;
  entry.size = tensor_allocation.buffer.GetSize();
  return entry;
}

void check_cast_tensor(
    const at::Tensor& tensor,
    const char* operation) {
  TORCH_CHECK(
      tensor.device().is_privateuseone(),
      operation,
      " expects a WebGPU tensor");
  TORCH_CHECK(
      tensor.layout() == at::kStrided,
      operation,
      " supports only strided tensors");
  TORCH_CHECK(
      tensor.scalar_type() == at::kInt ||
          tensor.scalar_type() == at::kLong ||
          tensor.scalar_type() == at::kFloat ||
          tensor.scalar_type() == at::kBool,
      operation,
      " supports torch.int32, range-limited torch.int64, torch.float32, and "
      "byte-packed torch.bool");
  TORCH_CHECK(
      !at::GradMode::is_enabled() || !tensor.requires_grad(),
      "WebGPU autograd is not supported; use `torch.no_grad()` for inference");
  TORCH_CHECK(
      tensor.dim() <= 8, operation, " supports at most 8 dimensions");
  TORCH_CHECK(
      tensor.storage_offset() >= 0,
      operation,
      " does not support negative storage offsets");
}

void validate_storage_span(
    const at::Tensor& tensor,
    std::uint32_t word_width,
    const char* operation) {
  for (const auto dim : c10::irange(tensor.dim())) {
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        operation,
        " does not support negative strides");
  }
  if (tensor.numel() == 0) {
    return;
  }
  std::uint64_t maximum_index =
      static_cast<std::uint64_t>(tensor.storage_offset());
  for (const auto dim : c10::irange(tensor.dim())) {
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
  if (word_width == 0) {
    checked_u32(
        static_cast<std::int64_t>(maximum_index),
        "Bool cast maximum storage index");
    TORCH_CHECK(
        maximum_index < cast_allocation(tensor).buffer.GetSize(),
        operation,
        " Bool view exceeds its GPUBuffer storage");
    return;
  }
  TORCH_CHECK(
      maximum_index <=
          (std::numeric_limits<std::uint32_t>::max() - (word_width - 1)) /
              word_width,
      operation,
      " storage index does not fit uint32 WebGPU metadata");
  const auto maximum_word = maximum_index * word_width + word_width - 1;
  TORCH_CHECK(
      maximum_word <
          cast_allocation(tensor).buffer.GetSize() / sizeof(std::uint32_t),
      operation,
      " view exceeds its GPUBuffer storage");
}

std::pair<std::uint32_t, std::uint32_t> dispatch_elements(
    std::uint32_t length) {
  constexpr std::uint32_t workgroup_size = 64;
  constexpr std::uint32_t max_workgroups = 65535;
  const auto groups = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(length) + workgroup_size - 1) /
      workgroup_size);
  const auto x = std::min(groups, max_workgroups);
  const auto y = (groups + x - 1) / x;
  TORCH_CHECK(y <= max_workgroups, "WebGPU cast dispatch is too large");
  return {x, y};
}

struct CastParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t source_offset;
  std::uint32_t destination_offset;
  std::uint32_t source_word_width;
  std::uint32_t destination_word_width;
  std::uint32_t convert_to_float;
  std::uint32_t dispatch_x;
  std::uint32_t sizes[8];
  std::uint32_t source_strides[8];
  std::uint32_t destination_strides[8];
};

static_assert(sizeof(CastParams) == 128);

struct BoolToLongParams {
  std::uint32_t length;
  std::uint32_t ndim;
  std::uint32_t source_offset;
  std::uint32_t dispatch_x;
  std::uint32_t sizes[8];
  std::uint32_t source_strides[8];
};

static_assert(sizeof(BoolToLongParams) == 80);

at::Tensor upload_cpu_tensor(
    const at::Tensor& input,
    at::ScalarType target_dtype,
    const at::Device& target_device,
    bool non_blocking,
    c10::MemoryFormat format) {
  TORCH_CHECK(
      input.device().is_cpu(),
      "WebGPU _to_copy upload expects a CPU tensor");
  TORCH_CHECK(
      input.layout() == at::kStrided,
      "CPU to WebGPU _to_copy supports only strided tensors");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat ||
          input.scalar_type() == at::kInt ||
          input.scalar_type() == at::kLong ||
          input.scalar_type() == at::kBool,
      "CPU to WebGPU _to_copy supports only torch.float32, torch.int32, "
      "signed-int32-valued torch.int64, and torch.bool");
  TORCH_CHECK(
      target_dtype == input.scalar_type(),
      "CPU to WebGPU _to_copy requires matching dtypes; convert on CPU "
      "before transferring to WebGPU");
  TORCH_CHECK(
      !at::GradMode::is_enabled() || !input.requires_grad(),
      "WebGPU autograd is not supported; use `torch.no_grad()` for inference");
  TORCH_CHECK(
      input.dim() <= 8,
      "CPU to WebGPU _to_copy supports at most 8 dimensions");
  TORCH_CHECK(
      input.storage_offset() >= 0,
      "CPU to WebGPU _to_copy does not support negative storage offsets");
  TORCH_CHECK(
      format == c10::MemoryFormat::Contiguous || input.is_contiguous(),
      "CPU to WebGPU _to_copy requires contiguous input for preserve_format; "
      "request contiguous_format or call contiguous() first");

  const auto options = input.options()
                           .dtype(target_dtype)
                           .layout(at::kStrided)
                           .device(target_device)
                           .pinned_memory(false);
  at::Tensor output;
  if (format == c10::MemoryFormat::Preserve &&
      input.is_non_overlapping_and_dense()) {
    output = at::empty_strided(input.sizes(), input.strides(), options);
  } else {
    output = at::empty(
        input.sizes(), options, c10::MemoryFormat::Contiguous);
  }

  // Route through the backend's established CPU -> WebGPU `_copy_from`
  // implementation. It owns byte-span checks, queue upload, and the canonical
  // signed-int32 validation for real 8-byte Long storage.
  output.copy_(input, non_blocking);
  return output;
}

at::Tensor to_copy_impl(
    const at::Tensor& input,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format) {
  TORCH_CHECK(
      !layout || *layout == at::kStrided,
      "WebGPU _to_copy supports only strided layout");
  TORCH_CHECK(
      !pin_memory.value_or(false), "WebGPU tensors cannot be pinned");
  const auto format = memory_format.value_or(c10::MemoryFormat::Preserve);
  TORCH_CHECK(
      format == c10::MemoryFormat::Preserve ||
          format == c10::MemoryFormat::Contiguous,
      "WebGPU _to_copy supports only preserve_format or contiguous_format");

  const auto target_dtype = dtype.value_or(input.scalar_type());
  if (input.device().is_cpu()) {
    TORCH_CHECK(
        device && device->is_privateuseone() && device->index() <= 0,
        "CPU to WebGPU _to_copy requires device='webgpu'");
    return upload_cpu_tensor(
        input,
        target_dtype,
        *device,
        non_blocking,
        format);
  }

  check_cast_tensor(input, "WebGPU _to_copy input");
  if (device && !device->is_privateuseone()) {
    TORCH_CHECK(
        !device->is_cpu(),
        "browser WebGPU readback is asynchronous; use "
        "`await torch.webgpu.to_cpu_async(tensor)` instead of `.cpu()`");
    TORCH_CHECK(false, "WebGPU _to_copy cannot convert to device ", *device);
  }
  TORCH_CHECK(
      !device || device->index() <= 0,
      "WebGPU _to_copy supports only device 0");

  const bool same_dtype = target_dtype == input.scalar_type();
  const bool convert_to_float =
      target_dtype == at::kFloat &&
      (input.scalar_type() == at::kInt || input.scalar_type() == at::kLong);
  const bool convert_bool_to_long =
      input.scalar_type() == at::kBool && target_dtype == at::kLong;
  const bool convert_float_to_long =
      input.scalar_type() == at::kFloat && target_dtype == at::kLong;
  TORCH_CHECK(
      same_dtype || convert_to_float || convert_bool_to_long ||
          convert_float_to_long,
      "WebGPU _to_copy supports only torch.bool to restricted torch.int64, "
      "torch.int32/torch.int64 to torch.float32, checked truncating "
      "torch.float32 to restricted torch.int64, or a same-dtype copy");

  at::Tensor output;
  if (!convert_bool_to_long && !convert_float_to_long &&
      format == c10::MemoryFormat::Preserve &&
      input.is_non_overlapping_and_dense()) {
    output = at::empty_strided(
        input.sizes(),
        input.strides(),
        input.options().dtype(target_dtype));
  } else {
    output = at::empty(
        input.sizes(),
        input.options().dtype(target_dtype),
        c10::MemoryFormat::Contiguous);
  }
  if (input.numel() == 0) {
    return output;
  }
  check_cast_tensor(output, "WebGPU _to_copy output");

  const std::uint32_t source_word_width = input.scalar_type() == at::kBool
      ? 0
      : (input.scalar_type() == at::kLong ? 2 : 1);
  const std::uint32_t destination_word_width = output.scalar_type() == at::kBool
      ? 0
      : (output.scalar_type() == at::kLong ? 2 : 1);
  validate_storage_span(input, source_word_width, "WebGPU _to_copy input");
  validate_storage_span(
      output, destination_word_width, "WebGPU _to_copy output");

  if (convert_bool_to_long) {
    TORCH_CHECK(
        output.is_contiguous() && output.storage_offset() == 0,
        "WebGPU Bool to Long conversion requires contiguous output");
    TORCH_CHECK(
        output.element_size() == 8,
        "WebGPU Bool to Long conversion requires ordinary 8-byte Long storage");

    BoolToLongParams params{};
    params.length = checked_u32(input.numel(), "Bool-to-Long element count");
    params.ndim = checked_u32(input.dim(), "Bool-to-Long input rank");
    params.source_offset =
        checked_u32(input.storage_offset(), "Bool-to-Long source offset");
    for (const auto dim : c10::irange(input.dim())) {
      params.sizes[dim] =
          checked_u32(input.size(dim), "Bool-to-Long input size");
      params.source_strides[dim] =
          checked_u32(input.stride(dim), "Bool-to-Long source stride");
    }
    const auto [dispatch_x, dispatch_y] = dispatch_elements(params.length);
    params.dispatch_x = dispatch_x;

    auto params_buffer =
        make_params_buffer("bool-to-Long params", params);
    auto entries = std::vector<wgpu::BindGroupEntry>{
        cast_tensor_entry(0, input),
        cast_tensor_entry(1, output),
        buffer_entry(2, params_buffer, sizeof(params))};
    dispatch(bool_to_long_kernel(), entries, dispatch_x, dispatch_y, 1);
    return output;
  }

  CastParams params{};
  params.length = checked_u32(input.numel(), "cast element count");
  params.ndim = checked_u32(input.dim(), "cast input rank");
  params.source_offset =
      checked_u32(input.storage_offset(), "cast source offset");
  params.destination_offset =
      checked_u32(output.storage_offset(), "cast destination offset");
  params.source_word_width = source_word_width;
  params.destination_word_width = destination_word_width;
  params.convert_to_float =
      convert_to_float ? 1u : (convert_float_to_long ? 2u : 0u);
  for (const auto dim : c10::irange(input.dim())) {
    params.sizes[dim] = checked_u32(input.size(dim), "cast input size");
    params.source_strides[dim] =
        checked_u32(input.stride(dim), "cast source stride");
    params.destination_strides[dim] =
        checked_u32(output.stride(dim), "cast destination stride");
  }
  const auto [dispatch_x, dispatch_y] = source_word_width == 0
      ? std::pair<std::uint32_t, std::uint32_t>{1, 1}
      : dispatch_elements(params.length);
  params.dispatch_x = dispatch_x;

  auto params_buffer = make_params_buffer("int-to-float params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      cast_tensor_entry(0, input),
      cast_tensor_entry(1, output),
      buffer_entry(2, params_buffer, sizeof(params))};
  dispatch(int_to_float_kernel(), entries, dispatch_x, dispatch_y, 1);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("_to_copy", TORCH_FN(to_copy_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
