from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NeTensorSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.boolean = (
            ROOT / "webgpu" / "llm_kernels" / "boolean.cpp"
        ).read_text(encoding="utf-8")
        cls.shader = (
            ROOT / "webgpu" / "llm_kernels" / "ne_tensor.wgsl"
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

    def test_exact_tensor_overload_has_one_compiled_registration(self) -> None:
        # Transformers 4.46.3 materializes pad_token_id as a rank-zero Long
        # tensor before calling inputs.ne(pad_token_id), selecting ne.Tensor.
        self.assertIn("transformers==4.46.3", self.requirements)
        self.assertEqual(self.project_cpp.count('module.impl("ne.Tensor"'), 1)
        self.assertNotIn('module.impl("ne.Scalar"', self.project_cpp)
        self.assertIn(
            'module.impl("ne.Tensor", TORCH_FN(ne_tensor));',
            self.boolean,
        )
        self.assertIn("llm_kernels/boolean.cpp", self.patch)
        self.assertNotIn("torch-webgpu/csrc/ops/comparison.cpp", self.patch)

    def test_integer_tensor_kernel_is_broadcast_and_device_safe(self) -> None:
        start = self.boolean.index("at::Tensor ne_tensor(")
        end = self.boolean.index("struct AllBoolParams", start)
        implementation = self.boolean[start:end]
        self.assertIn("at::kInt || input.scalar_type() == at::kLong", self.boolean)
        self.assertIn("lhs.device() == rhs.device()", implementation)
        self.assertIn("at::infer_size(lhs.sizes(), rhs.sizes())", implementation)
        self.assertIn("output_shape.size() <= 8", implementation)
        self.assertIn("validate_storage_span(lhs, operation);", implementation)
        self.assertIn("validate_storage_span(rhs, operation);", implementation)
        self.assertIn("lhs.scalar_type() == at::kLong ? 1u : 0u", implementation)
        self.assertIn("rhs.scalar_type() == at::kLong ? 2u : 0u", implementation)
        self.assertIn("dispatch_words(params.length)", implementation)
        self.assertNotIn("at::kCPU", implementation)
        self.assertNotIn(".cpu()", implementation)

    def test_shader_handles_strides_mixed_integer_kinds_and_packed_bool(self) -> None:
        self.assertIn("params.output_sizes[dim]", self.shader)
        self.assertIn("params.lhs_strides[input_dim]", self.shader)
        self.assertIn("params.rhs_strides[input_dim]", self.shader)
        self.assertIn("params.lhs_sizes[input_dim] != 1u", self.shader)
        self.assertIn("params.rhs_sizes[input_dim] != 1u", self.shader)
        self.assertIn("lhs[lhs_word + 1u] != rhs[rhs_word + 1u]", self.shader)
        self.assertIn("canonical_high(lhs_low)", self.shader)
        self.assertIn("canonical_high(rhs_low)", self.shader)
        self.assertIn("let output_word = group * 64u + local_id.x", self.shader)
        self.assertIn("let first_index = output_word * 4u", self.shader)
        self.assertIn("output[output_word] = packed", self.shader)

    def test_staging_and_both_runtime_gates_cover_ne_tensor(self) -> None:
        self.assertIn('"ne_tensor.wgsl": "kNeTensor"', self.staging)
        self.assertIn('project_kernels / "ne_tensor.wgsl"', self.staging)
        self.assertIn('"ne_tensor.wgsl"', self.raw_gate)
        self.assertIn('"ne.Tensor Long scalar broadcast"', self.raw_gate)
        self.assertIn('"ne.Tensor Int Long scalar"', self.raw_gate)
        self.assertIn('"ne.Tensor same-dtype Int broadcast"', self.raw_gate)
        self.assertIn("canonical tail", self.raw_gate)
        self.assertIn("output bound", self.raw_gate)
        self.assertIn("result.dispatches = 16", self.raw_gate)
        self.assertIn(
            "generation_long.ne(generation_pad_token)", self.functional_gate
        )
        self.assertIn(
            "generation_int.ne(generation_pad_token)", self.functional_gate
        )
        self.assertIn(
            "generation_elements.ne(generation_test)", self.functional_gate
        )


if __name__ == "__main__":
    unittest.main()
