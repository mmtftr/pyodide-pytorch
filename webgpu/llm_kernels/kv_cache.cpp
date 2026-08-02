#include "llm_common.h"

#include <algorithm>
#include <limits>

namespace pyodide_pytorch::webgpu::llm {
namespace {

constexpr std::uint32_t kMaxWorkgroupsPerDimension = 65535;

ComputeKernel& kv_cache_update_kernel() {
  static ComputeKernel kernel = make_kernel(
      "pyodide-pytorch fused indexed KV-cache update",
      shaders::kKvCacheUpdate,
      {wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::ReadOnlyStorage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Storage,
       wgpu::BufferBindingType::Uniform});
  return kernel;
}

struct KvCacheParams {
  std::uint32_t elements;
  std::uint32_t batch;
  std::uint32_t heads;
  std::uint32_t tokens;
  std::uint32_t head_dim;
  std::uint32_t capacity;
  std::uint32_t key_state_offset;
  std::uint32_t value_state_offset;
  std::uint32_t key_cache_offset;
  std::uint32_t value_cache_offset;
  std::uint32_t position_offset;
  std::uint32_t position_words;
  std::uint32_t dispatch_x;
  std::uint32_t padding[3];
  std::uint32_t key_state_strides[4];
  std::uint32_t value_state_strides[4];
  std::uint32_t key_cache_strides[4];
  std::uint32_t value_cache_strides[4];
};

static_assert(sizeof(KvCacheParams) == 128);

void set_positive_strides(
    std::uint32_t destination[4],
    const at::Tensor& tensor,
    const char* description) {
  TORCH_CHECK(tensor.dim() == 4, description, " expects a rank-4 tensor");
  for (const auto dim : c10::irange(4)) {
    TORCH_CHECK(
        tensor.stride(dim) >= 0,
        description,
        " does not support negative strides");
    destination[dim] = checked_u32(tensor.stride(dim), description);
  }
}

void check_distinct_write_buffer(
    const at::Tensor& writable,
    const at::Tensor& other,
    const char* operation) {
  if (writable.numel() == 0 || other.numel() == 0) {
    return;
  }
  TORCH_CHECK(
      allocation(writable).buffer.Get() != allocation(other).buffer.Get(),
      operation,
      " requires cache storage not to alias another operand");
}

void check_float_storage_span(
    const at::Tensor& tensor,
    const char* description) {
  if (tensor.numel() == 0) {
    return;
  }
  std::uint64_t maximum = checked_u32(tensor.storage_offset(), description);
  for (const auto dim : c10::irange(4)) {
    const auto size = checked_u32(tensor.size(dim), description);
    const auto stride = checked_u32(tensor.stride(dim), description);
    if (size <= 1) {
      continue;
    }
    const auto extent =
        static_cast<std::uint64_t>(size - 1) * stride;
    TORCH_CHECK(
        extent <= std::numeric_limits<std::uint32_t>::max() - maximum,
        description,
        " storage span does not fit uint32 WebGPU metadata");
    maximum += extent;
  }
  TORCH_CHECK(
      maximum < allocation(tensor).buffer.GetSize() / sizeof(float),
      description,
      " tensor exceeds its GPUBuffer storage");
}

void check_position_storage_span(
    const at::Tensor& positions,
    std::uint32_t words) {
  if (positions.numel() == 0) {
    return;
  }
  const auto offset = checked_u32(
      positions.storage_offset(), "indexed KV-cache position offset");
  const auto count = checked_u32(
      positions.numel(), "indexed KV-cache position count");
  const auto last_word =
      (static_cast<std::uint64_t>(offset) + count) * words - 1;
  TORCH_CHECK(
      last_word <= std::numeric_limits<std::uint32_t>::max(),
      "indexed KV-cache position storage span does not fit uint32 metadata");
  TORCH_CHECK(
      last_word < allocation(positions).buffer.GetSize() / sizeof(std::uint32_t),
      "indexed KV-cache positions exceed their GPUBuffer storage");
}

void update_kv_cache_impl(
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& key_states,
    const at::Tensor& value_states,
    const at::Tensor& cache_position) {
  constexpr const char* operation = "WebGPU indexed KV-cache update";
  check_inference_tensor(key_cache, operation, at::kFloat);
  check_inference_tensor(value_cache, operation, at::kFloat);
  check_inference_tensor(key_states, operation, at::kFloat);
  check_inference_tensor(value_states, operation, at::kFloat);
  check_inference_tensor(cache_position, operation);
  TORCH_CHECK(
      cache_position.scalar_type() == at::kInt ||
          cache_position.scalar_type() == at::kLong,
      operation,
      " positions must be torch.int32 or signed-int32-valued torch.int64");
  TORCH_CHECK(
      key_cache.device() == value_cache.device() &&
          key_cache.device() == key_states.device() &&
          key_cache.device() == value_states.device() &&
          key_cache.device() == cache_position.device(),
      operation,
      " requires every tensor on the same WebGPU device");
  TORCH_CHECK(
      key_cache.dim() == 4 && value_cache.dim() == 4 &&
          key_states.dim() == 4 && value_states.dim() == 4,
      operation,
      " expects [batch, heads, sequence, head_dim] tensors");
  TORCH_CHECK(
      key_cache.sizes() == value_cache.sizes(),
      operation,
      " requires matching key/value cache shapes");
  TORCH_CHECK(
      key_states.sizes() == value_states.sizes(),
      operation,
      " requires matching key/value state shapes");
  TORCH_CHECK(
      key_states.size(0) == key_cache.size(0) &&
          key_states.size(1) == key_cache.size(1) &&
          key_states.size(3) == key_cache.size(3),
      operation,
      " state and cache batch/head/feature dimensions must match");
  TORCH_CHECK(
      key_states.size(2) <= key_cache.size(2),
      operation,
      " state sequence exceeds cache capacity");
  TORCH_CHECK(
      cache_position.dim() == 1 &&
          cache_position.numel() == key_states.size(2) &&
          cache_position.is_contiguous(),
      operation,
      " positions must be a contiguous vector matching the state sequence");
  TORCH_CHECK(
      key_cache.is_contiguous() && value_cache.is_contiguous(),
      operation,
      " requires contiguous preallocated cache storage");

  check_distinct_write_buffer(key_cache, value_cache, operation);
  check_distinct_write_buffer(key_cache, key_states, operation);
  check_distinct_write_buffer(key_cache, value_states, operation);
  check_distinct_write_buffer(key_cache, cache_position, operation);
  check_distinct_write_buffer(value_cache, key_states, operation);
  check_distinct_write_buffer(value_cache, value_states, operation);
  check_distinct_write_buffer(value_cache, cache_position, operation);

  if (key_states.numel() == 0) {
    return;
  }

  check_float_storage_span(key_cache, "indexed KV-cache keys");
  check_float_storage_span(value_cache, "indexed KV-cache values");
  check_float_storage_span(key_states, "indexed KV-cache key states");
  check_float_storage_span(value_states, "indexed KV-cache value states");

  KvCacheParams params{};
  params.elements = checked_u32(
      key_states.numel(), "indexed KV-cache state element count");
  params.batch = checked_u32(key_states.size(0), "indexed KV-cache batch");
  params.heads = checked_u32(key_states.size(1), "indexed KV-cache heads");
  params.tokens = checked_u32(key_states.size(2), "indexed KV-cache tokens");
  params.head_dim =
      checked_u32(key_states.size(3), "indexed KV-cache head dimension");
  params.capacity =
      checked_u32(key_cache.size(2), "indexed KV-cache capacity");
  TORCH_CHECK(
      params.batch > 0 && params.heads > 0 && params.tokens > 0 &&
          params.head_dim > 0 && params.capacity > 0,
      operation,
      " requires nonempty dimensions");
  params.key_state_offset = checked_u32(
      key_states.storage_offset(), "indexed KV-cache key-state offset");
  params.value_state_offset = checked_u32(
      value_states.storage_offset(), "indexed KV-cache value-state offset");
  params.key_cache_offset = checked_u32(
      key_cache.storage_offset(), "indexed KV-cache key offset");
  params.value_cache_offset = checked_u32(
      value_cache.storage_offset(), "indexed KV-cache value offset");
  params.position_offset = checked_u32(
      cache_position.storage_offset(), "indexed KV-cache position offset");
  params.position_words = cache_position.scalar_type() == at::kLong ? 2 : 1;
  check_position_storage_span(cache_position, params.position_words);
  set_positive_strides(
      params.key_state_strides, key_states, "indexed KV-cache key states");
  set_positive_strides(
      params.value_state_strides,
      value_states,
      "indexed KV-cache value states");
  set_positive_strides(
      params.key_cache_strides, key_cache, "indexed KV-cache keys");
  set_positive_strides(
      params.value_cache_strides, value_cache, "indexed KV-cache values");

  const auto workgroups = (params.elements + 63) / 64;
  params.dispatch_x = std::min(workgroups, kMaxWorkgroupsPerDimension);
  const auto dispatch_y =
      (workgroups + params.dispatch_x - 1) / params.dispatch_x;
  TORCH_CHECK(
      dispatch_y <= kMaxWorkgroupsPerDimension,
      operation,
      " dispatch exceeds WebGPU limits");

  auto params_buffer =
      make_params_buffer("indexed KV-cache params", params);
  auto entries = std::vector<wgpu::BindGroupEntry>{
      tensor_entry(0, key_states),
      tensor_entry(1, value_states),
      tensor_entry(2, cache_position),
      tensor_entry(3, key_cache),
      tensor_entry(4, value_cache),
      buffer_entry(5, params_buffer, sizeof(params))};
  dispatch(
      kv_cache_update_kernel(), entries, params.dispatch_x, dispatch_y, 1);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(webgpu, module) {
  module.def(
      "update_kv_cache_(Tensor(a!) key_cache, Tensor(b!) value_cache, "
      "Tensor key_states, Tensor value_states, Tensor cache_position) -> ()");
}

TORCH_LIBRARY_IMPL(webgpu, PrivateUse1, module) {
  module.impl("update_kv_cache_", TORCH_FN(update_kv_cache_impl));
}

} // namespace pyodide_pytorch::webgpu::llm
