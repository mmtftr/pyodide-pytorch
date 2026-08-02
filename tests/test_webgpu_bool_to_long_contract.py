from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BoolToLongSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cpp = (
            ROOT / "webgpu" / "llm_kernels" / "type_conversion.cpp"
        ).read_text(encoding="utf-8")
        cls.shader = (
            ROOT / "webgpu" / "llm_kernels" / "bool_to_long.wgsl"
        ).read_text(encoding="utf-8")
        cls.project_cpp = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "webgpu" / "llm_kernels").glob("*.cpp")
        )
        cls.staging = (
            ROOT / "scripts" / "stage_webgpu_sources.py"
        ).read_text(encoding="utf-8")
        cls.patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0010-add-experimental-browser-webgpu-backend.patch"
        ).read_text(encoding="utf-8")
        cls.raw_gate = (
            ROOT / "tests" / "webgpu-generation-long-wgsl.mjs"
        ).read_text(encoding="utf-8")
        cls.functional_gate = (ROOT / "tests" / "webgpu.html").read_text(
            encoding="utf-8"
        )
        cls.requirements = (
            ROOT / "config" / "transformers-browser-requirements.txt"
        ).read_text(encoding="utf-8")

    def test_pinned_transformers_follow_on_uses_one_existing_registration(
        self,
    ) -> None:
        self.assertIn("transformers==4.46.3", self.requirements)
        self.assertIn(
            "generation_attention_mask = "
            "generation_long.ne(generation_pad_token).long()",
            self.functional_gate,
        )
        self.assertEqual(self.project_cpp.count('module.impl("_to_copy"'), 1)
        self.assertIn(
            'module.impl("_to_copy", TORCH_FN(to_copy_impl));', self.cpp
        )
        self.assertEqual(
            self.patch.count(
                "pyodide-pytorch-webgpu/llm_kernels/type_conversion.cpp"
            ),
            1,
        )
        self.assertNotIn(
            "torch-webgpu/csrc/ops/type_conversion.cpp", self.patch
        )

    def test_to_copy_matches_the_pinned_schema_signature(self) -> None:
        self.assertIn(
            """at::Tensor to_copy_impl(
    const at::Tensor& input,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format)""",
            self.cpp,
        )

    def test_bool_branch_is_gpu_only_and_preserves_existing_casts(self) -> None:
        start = self.cpp.index("const bool same_dtype")
        end = self.cpp.index("CastParams params{}", start)
        implementation = self.cpp[start:end]
        self.assertIn(
            "input.scalar_type() == at::kBool && target_dtype == at::kLong",
            implementation,
        )
        self.assertIn(
            "same_dtype || convert_to_float || convert_bool_to_long",
            implementation,
        )
        self.assertIn(
            "input.scalar_type() == at::kInt || input.scalar_type() == at::kLong",
            implementation,
        )
        self.assertIn(
            "!convert_bool_to_long && format == c10::MemoryFormat::Preserve",
            implementation,
        )
        self.assertIn(
            "output.is_contiguous() && output.storage_offset() == 0",
            implementation,
        )
        self.assertIn("output.element_size() == 8", implementation)
        self.assertIn(
            "dispatch(bool_to_long_kernel(), entries, dispatch_x, dispatch_y, 1)",
            implementation,
        )
        self.assertNotIn("at::kCPU", implementation)
        self.assertNotIn(".cpu()", implementation)

    def test_cpp_validates_strided_views_and_skips_empty_dispatch(self) -> None:
        self.assertIn("tensor.dim() <= 8", self.cpp)
        self.assertIn("tensor.storage_offset() >= 0", self.cpp)
        self.assertIn("tensor.stride(dim) >= 0", self.cpp)
        self.assertIn(
            'validate_storage_span(input, source_word_width, "WebGPU _to_copy input")',
            self.cpp,
        )
        self.assertIn("static_assert(sizeof(BoolToLongParams) == 80)", self.cpp)
        empty = self.cpp.index("if (input.numel() == 0)")
        dispatch = self.cpp.index("dispatch(bool_to_long_kernel()", empty)
        self.assertLess(empty, dispatch)

    def test_shader_canonicalizes_packed_bool_into_real_long_storage(self) -> None:
        self.assertIn("var<storage, read> source: array<u32>", self.shader)
        self.assertIn(
            "var<storage, read_write> destination: array<u32>", self.shader
        )
        self.assertIn(
            "workgroup_id.x + workgroup_id.y * params.dispatch_x", self.shader
        )
        self.assertIn("source_index += coordinate * source_stride_at(dim)", self.shader)
        self.assertIn("source[source_index >> 2u]", self.shader)
        self.assertIn("(source_index & 3u) * 8u", self.shader)
        self.assertIn(
            "select(0u, 1u, source_byte != 0u)", self.shader
        )
        self.assertIn("let destination_word = linear_index * 2u", self.shader)
        self.assertIn("destination[destination_word] = value", self.shader)
        self.assertIn("destination[destination_word + 1u] = 0u", self.shader)

    def test_staging_and_raw_adapter_gates_cover_edges(self) -> None:
        self.assertIn('"bool_to_long.wgsl": "kBoolToLong"', self.staging)
        self.assertIn(
            'project_kernels / "bool_to_long.wgsl"', self.staging
        )
        self.assertIn('"bool_to_long.wgsl"', self.raw_gate)
        self.assertIn('process.env.WEBGPU_ADAPTER ?? "swiftshader"', self.raw_gate)
        self.assertIn('["hardware", "swiftshader"]', self.raw_gate)
        self.assertIn('name: "Bool to Long scalar offset"', self.raw_gate)
        self.assertIn(
            'name: "Bool to Long rank-eight strided tail"', self.raw_gate
        )
        self.assertIn('name: "Bool to Long empty"', self.raw_gate)
        self.assertIn("canonical high word", self.raw_gate)
        self.assertIn("suffix bound high", self.raw_gate)
        self.assertIn(
            "result.dispatches = 16 + result.cases.boolToLong.dispatches",
            self.raw_gate,
        )

    def test_functional_gate_reads_back_contiguous_long_edges(self) -> None:
        for name in (
            "generation_attention_mask",
            "generation_strided_bool_long",
            "generation_scalar_bool_long",
            "generation_expanded_bool_long",
            "generation_empty_bool_long",
        ):
            self.assertIn(
                f"await torch.webgpu.to_cpu_async(\n    {name}\n)",
                self.functional_gate,
            )
        self.assertIn("generation_attention_mask.is_contiguous()", self.functional_gate)
        self.assertIn(
            "torch.ops.webgpu.buffer_nbytes(generation_attention_mask)",
            self.functional_gate,
        )
        self.assertIn(
            "generation_long_cpu.ne(0).long()", self.functional_gate
        )
        self.assertIn(
            "generation_any_negative.expand(2, 3).long()",
            self.functional_gate,
        )


if __name__ == "__main__":
    unittest.main()
