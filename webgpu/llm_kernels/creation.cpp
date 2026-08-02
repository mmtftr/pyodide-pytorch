#include "llm_common.h"

#include <c10/core/DefaultDtype.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pyodide_pytorch::webgpu::llm {
namespace {

ComputeKernel& arange_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch arange",
      shaders::kArange,
      {wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& fill_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch fill",
      shaders::kFill,
      {wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct DispatchShape {
  std::uint32_t x;
  std::uint32_t y;
};

DispatchShape dispatch_shape(std::uint32_t elements) {
  constexpr std::uint32_t max_workgroups = 65535;
  const auto workgroups =
      (static_cast<std::uint64_t>(elements) + 63) / 64;
  const auto x = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(workgroups, max_workgroups));
  const auto y = static_cast<std::uint32_t>((workgroups + x - 1) / x);
  TORCH_CHECK(y <= max_workgroups, "WebGPU creation dispatch is too large");
  return {x, y};
}

void check_factory_options(
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    const char* operation) {
  TORCH_CHECK(
      layout.value_or(at::Layout::Strided) == at::Layout::Strided,
      operation,
      " supports only strided layout");
  TORCH_CHECK(
      !pin_memory.value_or(false), operation, " does not support pinned memory");
  TORCH_CHECK(
      device && device->is_privateuseone() && device->index() <= 0,
      operation,
      " requires device='webgpu'");
}

std::int32_t scalar_to_i32(
    const at::Scalar& scalar,
    const char* description,
    bool allow_bool = false) {
  std::int64_t value;
  if (scalar.isIntegral(allow_bool)) {
    value = scalar.toLong();
  } else if (scalar.isFloatingPoint()) {
    const auto floating = scalar.toDouble();
    TORCH_CHECK(
        std::isfinite(floating) && std::trunc(floating) == floating,
        description,
        " must be an integer");
    TORCH_CHECK(
        floating >= static_cast<double>(std::numeric_limits<std::int32_t>::min()) &&
            floating <= static_cast<double>(std::numeric_limits<std::int32_t>::max()),
        description,
        " is outside the signed-int32 WebGPU Long profile");
    value = static_cast<std::int64_t>(floating);
  } else {
    TORCH_CHECK(false, description, " must be a real scalar");
  }
  TORCH_CHECK(
      value >= std::numeric_limits<std::int32_t>::min() &&
          value <= std::numeric_limits<std::int32_t>::max(),
      description,
      " is outside the signed-int32 WebGPU Long profile");
  return static_cast<std::int32_t>(value);
}

std::uint32_t bits(std::int32_t value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint32_t bits(float value) {
  std::uint32_t result;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct ArangeParams {
  std::uint32_t length;
  std::uint32_t dispatch_x;
  std::uint32_t output_offset;
  std::uint32_t output_words;
  std::uint32_t start_bits;
  std::uint32_t step_bits;
  std::uint32_t is_float;
  std::uint32_t padding;
};

static_assert(sizeof(ArangeParams) == 32);

at::Tensor arange_impl(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  check_factory_options(layout, device, pin_memory, "WebGPU arange");

  const auto all_integral = start.isIntegral(false) &&
      end.isIntegral(false) && step.isIntegral(false);
  const auto output_dtype = dtype.value_or(
      all_integral ? at::kLong : c10::get_default_dtype_as_scalartype());
  TORCH_CHECK(
      output_dtype == at::kFloat || output_dtype == at::kInt ||
          output_dtype == at::kLong,
      "WebGPU arange supports only torch.float32, torch.int32, and "
      "signed-int32-valued torch.int64");

  std::uint64_t length = 0;
  ArangeParams params{};
  params.output_words = output_dtype == at::kLong ? 2 : 1;
  if (output_dtype == at::kFloat) {
    TORCH_CHECK(
        (start.isIntegral(true) || start.isFloatingPoint()) &&
            (end.isIntegral(true) || end.isFloatingPoint()) &&
            (step.isIntegral(true) || step.isFloatingPoint()),
        "WebGPU arange requires real scalar bounds");
    const auto start_value = start.toFloat();
    const auto end_value = end.toFloat();
    const auto step_value = step.toFloat();
    TORCH_CHECK(
        std::isfinite(start_value) && std::isfinite(end_value) &&
            std::isfinite(step_value),
        "WebGPU arange requires finite float32 bounds and step");
    TORCH_CHECK(step_value != 0.0f, "arange: step must be nonzero");
    if ((step_value > 0.0f && start_value < end_value) ||
        (step_value < 0.0f && start_value > end_value)) {
      const auto count = std::ceil(
          (static_cast<double>(end_value) - start_value) / step_value);
      TORCH_CHECK(
          std::isfinite(count) && count >= 0.0 &&
              count <= std::numeric_limits<std::uint32_t>::max(),
          "WebGPU arange output length is unsupported");
      length = static_cast<std::uint64_t>(count);
    }
    params.start_bits = bits(start_value);
    params.step_bits = bits(step_value);
    params.is_float = 1;
  } else {
    const auto start_value = scalar_to_i32(start, "arange start");
    const auto end_value = scalar_to_i32(end, "arange end");
    const auto step_value = scalar_to_i32(step, "arange step");
    TORCH_CHECK(step_value != 0, "arange: step must be nonzero");
    const auto start_wide = static_cast<std::int64_t>(start_value);
    const auto end_wide = static_cast<std::int64_t>(end_value);
    const auto step_wide = static_cast<std::int64_t>(step_value);
    if (step_wide > 0 && start_wide < end_wide) {
      length = 1 + static_cast<std::uint64_t>(
          (end_wide - 1 - start_wide) / step_wide);
    } else if (step_wide < 0 && start_wide > end_wide) {
      length = 1 + static_cast<std::uint64_t>(
          (start_wide - 1 - end_wide) / -step_wide);
    }
    TORCH_CHECK(
        length <= std::numeric_limits<std::uint32_t>::max(),
        "WebGPU arange output length is unsupported");
    if (length != 0) {
      const auto last = start_wide +
          static_cast<std::int64_t>(length - 1) * step_wide;
      TORCH_CHECK(
          last >= std::numeric_limits<std::int32_t>::min() &&
              last <= std::numeric_limits<std::int32_t>::max(),
          "WebGPU arange values exceed the signed-int32 Long profile");
    }
    params.start_bits = bits(start_value);
    params.step_bits = bits(step_value);
  }

  auto output = at::empty(
      {static_cast<std::int64_t>(length)},
      at::TensorOptions().dtype(output_dtype).device(*device));
  if (length == 0) {
    return output;
  }

  params.length = static_cast<std::uint32_t>(length);
  params.output_offset =
      checked_u32(output.storage_offset(), "arange output offset");
  const auto shape = dispatch_shape(params.length);
  params.dispatch_x = shape.x;
  auto params_buffer = make_params_buffer("arange params", params);
  dispatch(
      arange_kernel(),
      {tensor_entry(0, output), buffer_entry(1, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return output;
}

at::Tensor arange_end(
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return arange_impl(0, end, 1, dtype, layout, device, pin_memory);
}

at::Tensor arange_start(
    const at::Scalar& start,
    const at::Scalar& end,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return arange_impl(start, end, 1, dtype, layout, device, pin_memory);
}

struct FillParams {
  std::uint32_t length;
  std::uint32_t dispatch_x;
  std::uint32_t output_offset;
  std::uint32_t output_words;
  std::uint32_t low_word;
  std::uint32_t high_word;
  std::uint32_t padding[2];
};

static_assert(sizeof(FillParams) == 32);

at::Tensor& fill_scalar(at::Tensor& self, const at::Scalar& value) {
  check_inference_tensor(self, "WebGPU fill");
  TORCH_CHECK(self.is_contiguous(), "WebGPU fill requires contiguous output");
  if (self.numel() == 0) {
    return self;
  }

  FillParams params{};
  params.output_offset = checked_u32(self.storage_offset(), "fill output offset");
  if (self.scalar_type() == at::kBool) {
    const auto logical_length = checked_u32(self.numel(), "Bool fill element count");
    TORCH_CHECK(
        logical_length <=
            std::numeric_limits<std::uint32_t>::max() - params.output_offset,
        "Bool fill byte range does not fit uint32 WebGPU metadata");
    // output_words == 0 selects byte-packed Bool in the shader. One invocation
    // owns one destination word, including boundary-byte read/modify/write.
    params.output_words = 0;
    params.low_word = value.toBool() ? 1u : 0u;
    params.high_word = logical_length;
    params.length = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(params.output_offset & 3u) +
         logical_length + 3u) /
        4u);
  } else if (self.scalar_type() == at::kFloat) {
    params.length = checked_u32(self.numel(), "fill element count");
    params.output_words = 1;
    TORCH_CHECK(
        value.isIntegral(true) || value.isFloatingPoint(),
        "WebGPU float32 fill requires a real scalar");
    params.low_word = bits(value.toFloat());
  } else {
    params.length = checked_u32(self.numel(), "fill element count");
    params.output_words = self.scalar_type() == at::kLong ? 2 : 1;
    const auto integer = scalar_to_i32(value, "fill value", true);
    params.low_word = bits(integer);
    params.high_word = integer < 0 ? 0xffffffffu : 0u;
  }

  const auto shape = dispatch_shape(params.length);
  params.dispatch_x = shape.x;
  auto params_buffer = make_params_buffer("fill params", params);
  dispatch(
      fill_kernel(),
      {tensor_entry(0, self), buffer_entry(1, params_buffer, sizeof(params))},
      shape.x,
      shape.y);
  return self;
}

at::Tensor& zero_(at::Tensor& self) {
  return fill_scalar(self, 0);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("arange", TORCH_FN(arange_end));
  module.impl("arange.start", TORCH_FN(arange_start));
  module.impl("arange.start_step", TORCH_FN(arange_impl));
  module.impl("fill_.Scalar", TORCH_FN(fill_scalar));
  module.impl("zero_", TORCH_FN(zero_));
}

} // namespace pyodide_pytorch::webgpu::llm
