#include "llm_common.h"

#include <array>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaxWorkgroupsPerDimension = 65535;
constexpr std::uint32_t kLinearTile = 8;
constexpr std::uint32_t kLinearGemvColumns = 64;
constexpr std::uint32_t kLinearGemvSubgroupS4Columns = 4;
constexpr std::uint32_t kLinearGemmRows = 16;
constexpr std::uint32_t kLinearGemmColumns = 32;
constexpr std::uint32_t kLinearGemmInner = 16;

std::uint32_t divide_rounding_up(
    std::uint32_t value,
    std::uint32_t divisor) {
  return value / divisor + static_cast<std::uint32_t>(value % divisor != 0);
}

#if defined(__EMSCRIPTEN__)
EM_JS(int, linear_gemv_subgroup_s4_supported, (), {
  const state = globalThis.__torchWebGPU;
  if (!state || !state.device || state.linearGemvVariant === "portable") {
    return 0;
  }
  const device = state.device;
  const info = device.adapterInfo;
  return device.features && device.features.has("subgroups") && info &&
      info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
});
#else
int linear_gemv_subgroup_s4_supported() {
  return 0;
}
#endif

ComputeKernel& bmm_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch bmm",
      shaders::kBmm,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& linear_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch fused linear",
      shaders::kLinear,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& linear_gemv_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch contiguous linear gemv",
      shaders::kLinearGemv,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& linear_gemv_subgroup_s4_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch contiguous linear gemv subgroup s4",
      shaders::kLinearGemvSubgroupS4,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

ComputeKernel& linear_gemm_16x32_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch contiguous linear gemm 16x32",
      shaders::kLinearGemm16x32,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct BmmParams {
  std::uint32_t batch;
  std::uint32_t rows;
  std::uint32_t columns;
  std::uint32_t inner;
  std::uint32_t lhs_offset;
  std::uint32_t rhs_offset;
  std::uint32_t output_offset;
  std::uint32_t padding;
  std::uint32_t lhs_strides[4];
  std::uint32_t rhs_strides[4];
  std::uint32_t output_strides[4];
};

static_assert(sizeof(BmmParams) == 80);

struct LinearParams {
  std::uint32_t rows;
  std::uint32_t columns;
  std::uint32_t inner;
  std::uint32_t input_offset;
  std::uint32_t weight_offset;
  std::uint32_t bias_offset;
  std::uint32_t output_offset;
  std::uint32_t has_bias;
  std::uint32_t input_stride0;
  std::uint32_t input_stride1;
  std::uint32_t weight_stride0;
  std::uint32_t weight_stride1;
  std::uint32_t padding[4];
};

static_assert(sizeof(LinearParams) == 64);

bool is_contiguous_matrix(
    const at::Tensor& tensor,
    std::int64_t rows,
    std::int64_t columns) {
  return tensor.is_contiguous() && tensor.dim() == 2 &&
      tensor.sizes() == at::IntArrayRef({rows, columns}) &&
      tensor.stride(0) == columns && tensor.stride(1) == 1;
}

at::Tensor& bmm_out_impl(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    at::Tensor& output) {
  check_inference_tensor(lhs, "WebGPU bmm lhs", at::kFloat);
  check_inference_tensor(rhs, "WebGPU bmm rhs", at::kFloat);
  check_inference_tensor(output, "WebGPU bmm output", at::kFloat);
  TORCH_CHECK(lhs.dim() == 3 && rhs.dim() == 3, "WebGPU bmm expects 3-D inputs");
  TORCH_CHECK(lhs.size(0) == rhs.size(0), "WebGPU bmm batch mismatch");
  TORCH_CHECK(lhs.size(2) == rhs.size(1), "WebGPU bmm inner-dimension mismatch");
  TORCH_CHECK(
      output.sizes() == at::IntArrayRef({lhs.size(0), lhs.size(1), rhs.size(2)}),
      "WebGPU bmm output shape mismatch");
  TORCH_CHECK(output.is_contiguous(), "WebGPU bmm output must be contiguous");
  if (output.numel() == 0) {
    return output;
  }

  BmmParams params{};
  params.batch = checked_u32(lhs.size(0), "bmm batch");
  params.rows = checked_u32(lhs.size(1), "bmm rows");
  params.columns = checked_u32(rhs.size(2), "bmm columns");
  params.inner = checked_u32(lhs.size(2), "bmm inner dimension");
  params.lhs_offset = checked_u32(lhs.storage_offset(), "bmm lhs offset");
  params.rhs_offset = checked_u32(rhs.storage_offset(), "bmm rhs offset");
  params.output_offset =
      checked_u32(output.storage_offset(), "bmm output offset");
  for (const auto dim : c10::irange(3)) {
    params.lhs_strides[dim] = checked_u32(lhs.stride(dim), "bmm lhs stride");
    params.rhs_strides[dim] = checked_u32(rhs.stride(dim), "bmm rhs stride");
    params.output_strides[dim] =
        checked_u32(output.stride(dim), "bmm output stride");
  }
  TORCH_CHECK(params.batch <= 65535, "WebGPU bmm batch exceeds dispatch limit");

  auto params_buffer = make_params_buffer("bmm params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, lhs),
      tensor_entry(1, rhs),
      tensor_entry(2, output),
      buffer_entry(3, params_buffer, sizeof(params))};
  dispatch(
      bmm_kernel(),
      entries,
      (params.columns + 7) / 8,
      (params.rows + 7) / 8,
      params.batch);
  return output;
}

at::Tensor bmm_impl(const at::Tensor& lhs, const at::Tensor& rhs) {
  check_inference_tensor(lhs, "WebGPU bmm lhs", at::kFloat);
  check_inference_tensor(rhs, "WebGPU bmm rhs", at::kFloat);
  TORCH_CHECK(lhs.dim() == 3 && rhs.dim() == 3, "WebGPU bmm expects 3-D inputs");
  auto output = at::empty(
      {lhs.size(0), lhs.size(1), rhs.size(2)}, lhs.options());
  return bmm_out_impl(lhs, rhs, output);
}

at::Tensor matmul_impl(const at::Tensor& lhs, const at::Tensor& rhs) {
  check_inference_tensor(lhs, "WebGPU matmul lhs", at::kFloat);
  check_inference_tensor(rhs, "WebGPU matmul rhs", at::kFloat);
  TORCH_CHECK(lhs.dim() >= 2 && rhs.dim() >= 2,
      "WebGPU matmul currently expects matrix dimensions");
  if (lhs.dim() == 2 && rhs.dim() == 2) {
    return at::mm(lhs, rhs);
  }
  if (lhs.dim() == 3 && rhs.dim() == 3) {
    return bmm_impl(lhs, rhs);
  }
  TORCH_CHECK(lhs.dim() == rhs.dim(),
      "WebGPU batched matmul requires equal ranks");
  TORCH_CHECK(lhs.size(-1) == rhs.size(-2),
      "WebGPU batched matmul inner-dimension mismatch");
  for (const auto dim : c10::irange(lhs.dim() - 2)) {
    TORCH_CHECK(lhs.size(dim) == rhs.size(dim),
        "WebGPU batched matmul does not yet broadcast batch dimensions");
  }

  const auto batch_shape = lhs.sizes().slice(0, lhs.dim() - 2);
  const auto batch = product(batch_shape);
  auto lhs_contiguous = lhs.is_contiguous() ? lhs : lhs.contiguous();
  auto rhs_contiguous = rhs.is_contiguous() ? rhs : rhs.contiguous();
  auto lhs_3d = lhs_contiguous.reshape({batch, lhs.size(-2), lhs.size(-1)});
  auto rhs_3d = rhs_contiguous.reshape({batch, rhs.size(-2), rhs.size(-1)});
  auto output_3d = bmm_impl(lhs_3d, rhs_3d);
  auto output_shape = batch_shape.vec();
  output_shape.push_back(lhs.size(-2));
  output_shape.push_back(rhs.size(-1));
  return output_3d.reshape(output_shape);
}

at::Tensor& matmul_out_impl(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    at::Tensor& output) {
  auto result = matmul_impl(lhs, rhs);
  output.copy_(result);
  return output;
}

at::Tensor linear_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  check_inference_tensor(input, "WebGPU linear input", at::kFloat);
  check_inference_tensor(weight, "WebGPU linear weight", at::kFloat);
  TORCH_CHECK(input.dim() >= 1 && weight.dim() == 2,
      "WebGPU linear expects input rank >= 1 and a 2-D weight");
  TORCH_CHECK(input.size(-1) == weight.size(1),
      "WebGPU linear feature mismatch");
  auto contiguous_input = input.is_contiguous() ? input : input.contiguous();
  const auto rows = product(input.sizes().slice(0, input.dim() - 1));
  auto input_2d = contiguous_input.reshape({rows, input.size(-1)});
  auto output_2d = at::empty({rows, weight.size(0)}, input.options());

  at::Tensor contiguous_bias;
  if (bias && bias->defined()) {
    check_inference_tensor(*bias, "WebGPU linear bias", at::kFloat);
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == weight.size(0),
        "WebGPU linear bias shape mismatch");
    contiguous_bias = bias->is_contiguous() ? *bias : bias->contiguous();
  }
  if (output_2d.numel() != 0) {
    LinearParams params{};
    params.rows = checked_u32(rows, "linear row count");
    params.columns = checked_u32(weight.size(0), "linear output features");
    params.inner = checked_u32(input.size(-1), "linear input features");
    params.input_offset =
        checked_u32(input_2d.storage_offset(), "linear input offset");
    params.weight_offset =
        checked_u32(weight.storage_offset(), "linear weight offset");
    params.bias_offset = contiguous_bias.defined()
        ? checked_u32(contiguous_bias.storage_offset(), "linear bias offset")
        : 0;
    params.output_offset =
        checked_u32(output_2d.storage_offset(), "linear output offset");
    params.has_bias = contiguous_bias.defined() ? 1 : 0;
    params.input_stride0 =
        checked_u32(input_2d.stride(0), "linear input stride 0");
    params.input_stride1 =
        checked_u32(input_2d.stride(1), "linear input stride 1");
    params.weight_stride0 =
        checked_u32(weight.stride(0), "linear weight stride 0");
    params.weight_stride1 =
        checked_u32(weight.stride(1), "linear weight stride 1");

    at::Tensor empty_feature_binding;
    if (input_2d.numel() == 0) {
      // Empty WebGPU tensors intentionally have no GPUBuffer. The shader does
      // not read either matrix when inner == 0, but WebGPU still requires a
      // valid storage binding for every declared resource.
      empty_feature_binding = at::empty({1}, input.options());
    }
    const auto& input_binding =
        input_2d.numel() == 0 ? empty_feature_binding : input_2d;
    const auto& weight_binding =
        weight.numel() == 0 ? empty_feature_binding : weight;
    const auto& bias_binding = contiguous_bias.defined()
        ? contiguous_bias
        : input_binding;
    auto params_buffer = make_params_buffer("linear params", params);
    auto entries = std::vector<wgpu::BindGroupEntry>{
        tensor_entry(0, input_binding),
        tensor_entry(1, weight_binding),
        tensor_entry(2, bias_binding),
        tensor_entry(3, output_2d),
        buffer_entry(4, params_buffer, sizeof(params))};

    const auto input_is_contiguous = is_contiguous_matrix(
        input_2d, rows, input.size(-1));
    const auto weight_is_contiguous = is_contiguous_matrix(
        weight, weight.size(0), weight.size(1));
    const auto output_is_contiguous = is_contiguous_matrix(
        output_2d, rows, weight.size(0));
    const auto contiguous_fast_path =
        input_is_contiguous && weight_is_contiguous && output_is_contiguous;

    // The subgroup GEMV is restricted to devices that enabled subgroups and
    // report a fixed SIMD width of 32. Its S4 mapping handles N/K tails and
    // storage offsets. All other M == 1 shapes retain the portable GEMV.
    // GEMM deliberately omits bounds checks, so only exact contiguous tiles
    // may select it. Every strided or remaining tail shape stays generic.
    if (contiguous_fast_path && params.rows == 1) {
      if (linear_gemv_subgroup_s4_supported()) {
        const auto dispatch_x = divide_rounding_up(
            params.columns, kLinearGemvSubgroupS4Columns);
        TORCH_CHECK(dispatch_x <= kMaxWorkgroupsPerDimension,
            "WebGPU linear subgroup GEMV dispatch exceeds WebGPU limits");
        dispatch(
            linear_gemv_subgroup_s4_kernel(), entries, dispatch_x, 1, 1);
      } else {
        const auto dispatch_x =
            divide_rounding_up(params.columns, kLinearGemvColumns);
        TORCH_CHECK(dispatch_x <= kMaxWorkgroupsPerDimension,
            "WebGPU linear GEMV dispatch exceeds WebGPU limits");
        dispatch(linear_gemv_kernel(), entries, dispatch_x, 1, 1);
      }
    } else if (contiguous_fast_path &&
               params.rows % kLinearGemmRows == 0 &&
               params.columns % kLinearGemmColumns == 0 &&
               params.inner % kLinearGemmInner == 0) {
      const auto dispatch_x = params.columns / kLinearGemmColumns;
      const auto dispatch_y = params.rows / kLinearGemmRows;
      TORCH_CHECK(
          dispatch_x <= kMaxWorkgroupsPerDimension &&
              dispatch_y <= kMaxWorkgroupsPerDimension,
          "WebGPU linear GEMM dispatch dimensions exceed WebGPU limits");
      dispatch(
          linear_gemm_16x32_kernel(), entries, dispatch_x, dispatch_y, 1);
    } else {
      const auto dispatch_x = divide_rounding_up(params.columns, kLinearTile);
      const auto dispatch_y = divide_rounding_up(params.rows, kLinearTile);
      TORCH_CHECK(
          dispatch_x <= kMaxWorkgroupsPerDimension &&
              dispatch_y <= kMaxWorkgroupsPerDimension,
          "WebGPU linear dispatch dimensions exceed WebGPU limits");
      dispatch(
          linear_kernel(), entries, dispatch_x, dispatch_y, 1);
    }
  }
  auto output_shape = input.sizes().vec();
  output_shape.back() = weight.size(0);
  return output_2d.reshape(output_shape);
}

at::Tensor& linear_out_impl(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    at::Tensor& output) {
  auto result = linear_impl(input, weight, bias);
  output.copy_(result);
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, module) {
  module.impl("bmm", TORCH_FN(bmm_impl));
  module.impl("bmm.out", TORCH_FN(bmm_out_impl));
  module.impl("matmul", TORCH_FN(matmul_impl));
  module.impl("matmul.out", TORCH_FN(matmul_out_impl));
  module.impl("linear", TORCH_FN(linear_impl));
  module.impl("linear.out", TORCH_FN(linear_out_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
