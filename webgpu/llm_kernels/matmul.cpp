#include "llm_common.h"

#include <array>

namespace pyodide_pytorch::webgpu::llm {
namespace {

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
  const auto batch = input.numel() / input.size(-1);
  auto output_2d = at::mm(
      contiguous_input.reshape({batch, input.size(-1)}), weight.t());
  if (bias && bias->defined()) {
    check_inference_tensor(*bias, "WebGPU linear bias", at::kFloat);
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == weight.size(0),
        "WebGPU linear bias shape mismatch");
    output_2d = output_2d + *bias;
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
