from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GenerationControlSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cpp = (
            ROOT / "webgpu" / "llm_kernels" / "generation_control.cpp"
        ).read_text(encoding="utf-8")
        cls.shader = (
            ROOT / "webgpu" / "llm_kernels" / "bitwise_not_bool.wgsl"
        ).read_text(encoding="utf-8")
        cls.mul_shader = (
            ROOT / "webgpu" / "llm_kernels" / "mul_bool_tensor.wgsl"
        ).read_text(encoding="utf-8")
        cls.header = (
            ROOT / "webgpu" / "llm_kernels" / "generation_control.h"
        ).read_text(encoding="utf-8")
        cls.staging = (ROOT / "scripts" / "stage_webgpu_sources.py").read_text(
            encoding="utf-8"
        )
        cls.patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0010-add-experimental-browser-webgpu-backend.patch"
        ).read_text(encoding="utf-8")
        cls.raw_gate = (
            ROOT / "tests" / "webgpu-generation-long-wgsl.mjs"
        ).read_text(encoding="utf-8")

    def test_registration_is_bool_only_and_has_no_cpu_fallback(self) -> None:
        start = self.cpp.index("at::Tensor bitwise_not_bool(")
        end = self.cpp.index("std::int32_t scalar_to_i32(", start)
        implementation = self.cpp[start:end]
        self.assertIn(
            'check_strided_tensor(input, operation, at::kBool);',
            implementation,
        )
        self.assertIn("validate_storage_span(input, operation);", implementation)
        self.assertIn("dispatch_words(params.length, operation)", implementation)
        self.assertNotIn("at::kCPU", implementation)
        self.assertNotIn(".cpu()", implementation)
        self.assertIn(
            'module.impl("bitwise_not", TORCH_FN(bitwise_not_bool));',
            self.cpp,
        )
        self.assertNotIn('module.impl("bitwise_not.out"', self.cpp)

    def test_shader_owns_output_words_and_canonicalizes_bool_bytes(self) -> None:
        self.assertIn("var<storage, read> input: array<u32>", self.shader)
        self.assertIn("var<storage, read_write> output: array<u32>", self.shader)
        self.assertIn("let output_word = group * 64u + local_id.x", self.shader)
        self.assertIn("let first_index = output_word * 4u", self.shader)
        self.assertIn("(word >> shift) & 0xffu", self.shader)
        self.assertIn("!bool_at(input_index(linear_index))", self.shader)
        self.assertIn("output[output_word] = packed", self.shader)

    def test_staging_and_raw_gate_cover_the_exact_kernel(self) -> None:
        self.assertIn(
            '"bitwise_not_bool.wgsl": "kBitwiseNotBool"', self.staging
        )
        self.assertIn(
            'project_kernels / "bitwise_not_bool.wgsl"', self.staging
        )
        self.assertIn(
            "llm_kernels/generation_control.cpp", self.patch
        )
        self.assertIn('"bitwise_not_bool.wgsl"', self.raw_gate)
        self.assertIn('name: "bitwise_not scalar Bool"', self.raw_gate)
        self.assertIn('name: "bitwise_not strided Bool"', self.raw_gate)
        self.assertIn("value === 0 ? 1 : 0", self.raw_gate)

    def test_mul_tensor_has_one_registration_and_preserves_float_path(self) -> None:
        self.assertEqual(self.patch.count('module.impl("mul.Tensor"'), 1)
        self.assertEqual(self.patch.count('module.impl("mul.out"'), 1)
        self.assertNotIn('module.impl("mul.Tensor"', self.cpp)
        self.assertIn("at::Tensor mul_bool_tensor(", self.cpp)
        self.assertIn("at::Tensor mul_bool_tensor(", self.header)

        start = self.patch.index("+at::Tensor mul_tensor(")
        end = self.patch.index("+at::Tensor& mul_out(", start)
        implementation = self.patch[start:end]
        self.assertIn(
            "lhs.scalar_type() == at::kBool || rhs.scalar_type() == at::kBool",
            implementation,
        )
        self.assertIn(
            "pyodide_pytorch::webgpu::llm::mul_bool_tensor(lhs, rhs)",
            implementation,
        )
        self.assertIn("return binary_tensor<BinaryOp::Mul>(lhs, rhs);", implementation)

        out_start = end
        out_end = self.patch.index("+at::Tensor sub_tensor(", out_start)
        out_implementation = self.patch[out_start:out_end]
        self.assertNotIn("at::kBool", out_implementation)
        self.assertIn(
            "return binary_out<BinaryOp::Mul>(lhs, rhs, 1, output);",
            out_implementation,
        )

    def test_mul_bool_tensor_validates_and_broadcasts_strided_inputs(self) -> None:
        start = self.cpp.index("at::Tensor mul_bool_tensor_impl(")
        end = self.cpp.index("std::int32_t scalar_to_i32(", start)
        implementation = self.cpp[start:end]
        self.assertEqual(
            implementation.count(
                "check_strided_tensor(",
            ),
            2,
        )
        self.assertIn("at::infer_size(lhs.sizes(), rhs.sizes())", implementation)
        self.assertIn("output_shape.size() <= kMaximumDimensions", implementation)
        self.assertIn("lhs.device() == rhs.device()", implementation)
        self.assertIn("validate_storage_span(lhs, operation);", implementation)
        self.assertIn("validate_storage_span(rhs, operation);", implementation)
        self.assertIn("dispatch_words(params.length, operation)", implementation)
        self.assertNotIn("at::kCPU", implementation)
        self.assertNotIn(".cpu()", implementation)

    def test_mul_shader_owns_words_broadcasts_and_canonicalizes(self) -> None:
        self.assertIn("var<storage, read> lhs: array<u32>", self.mul_shader)
        self.assertIn("var<storage, read> rhs: array<u32>", self.mul_shader)
        self.assertIn("var<storage, read_write> output: array<u32>", self.mul_shader)
        self.assertIn("dim + params.lhs_ndim >= params.ndim", self.mul_shader)
        self.assertIn("params.lhs_sizes[input_dim] != 1u", self.mul_shader)
        self.assertIn("dim + params.rhs_ndim >= params.ndim", self.mul_shader)
        self.assertIn("params.rhs_sizes[input_dim] != 1u", self.mul_shader)
        self.assertIn("(word >> shift) & 0xffu", self.mul_shader)
        self.assertIn("bool_at(&lhs, lhs_index(linear_index)) &&", self.mul_shader)
        self.assertIn("output[output_word] = packed", self.mul_shader)

    def test_mul_staging_and_raw_gate_cover_scalar_broadcast_and_bounds(self) -> None:
        self.assertIn('"mul_bool_tensor.wgsl": "kMulBoolTensor"', self.staging)
        self.assertIn('project_kernels / "generation_control.h"', self.staging)
        self.assertIn('project_kernels / "mul_bool_tensor.wgsl"', self.staging)
        self.assertIn('"mul_bool_tensor.wgsl"', self.raw_gate)
        self.assertIn('"mul.Tensor scalar Bool false false"', self.raw_gate)
        self.assertIn('"mul.Tensor scalar Bool true true"', self.raw_gate)
        self.assertIn('"mul.Tensor broadcast strided Bool"', self.raw_gate)
        self.assertIn("mul.Tensor canonical tail", self.raw_gate)
        self.assertIn("mul.Tensor output bound", self.raw_gate)


if __name__ == "__main__":
    unittest.main()
