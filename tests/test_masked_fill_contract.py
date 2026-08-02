from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage_webgpu_sources  # noqa: E402


SOURCE = ROOT / "webgpu" / "llm_kernels" / "masked_fill.cpp"
SHADER = ROOT / "webgpu" / "llm_kernels" / "masked_fill_scalar.wgsl"
MASKING_SOURCE = ROOT / "webgpu" / "llm_kernels" / "masking.cpp"
RAW_GATE = ROOT / "tests" / "webgpu-masked-fill-wgsl.mjs"
PATCH = ROOT / "patches" / "pytorch" / "0010-add-experimental-browser-webgpu-backend.patch"
WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"
FIXTURE = ROOT / "tests" / "fixtures" / "transformers_tiny.json"


def cpp_struct_fields(source: str, name: str) -> list[str]:
    match = re.search(rf"struct {name} \{{(.*?)\n\}};", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"missing C++ struct {name}")
    return re.findall(r"std::uint32_t\s+(\w+)(?:\[\d+\])?;", match.group(1))


def wgsl_struct_fields(source: str, name: str) -> list[str]:
    match = re.search(rf"struct {name} \{{(.*?)\n\}};", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"missing WGSL struct {name}")
    return re.findall(r"^\s*(\w+):\s*[^,]+,\s*$", match.group(1), re.MULTILINE)


class MaskedFillContractTests(unittest.TestCase):
    def test_cpp_and_wgsl_uniform_abis_match(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        shader = SHADER.read_text(encoding="utf-8")
        metadata = [
            "output_sizes0",
            "output_sizes1",
            "self_sizes0",
            "self_sizes1",
            "self_strides0",
            "self_strides1",
            "mask_sizes0",
            "mask_sizes1",
            "mask_strides0",
            "mask_strides1",
        ]
        header = [
            "length",
            "ndim",
            "self_ndim",
            "mask_ndim",
            "self_offset",
            "mask_offset",
            "output_offset",
            "value_bits",
            "dispatch_x",
        ]
        self.assertEqual(
            cpp_struct_fields(source, "MaskedFillScalarParams"),
            header + ["padding"] + metadata,
        )
        self.assertEqual(
            wgsl_struct_fields(shader, "Params"),
            header + ["_pad0", "_pad1", "_pad2"] + metadata,
        )
        self.assertIn(
            "static_assert(sizeof(MaskedFillScalarParams) == 208);", source
        )

    def test_shader_has_one_float_writer_and_only_reads_packed_bool(self) -> None:
        shader = SHADER.read_text(encoding="utf-8")
        self.assertIn(
            "@binding(0) var<storage, read> self_values: array<f32>;", shader
        )
        self.assertIn(
            "@binding(1) var<storage, read> mask_values: array<u32>;", shader
        )
        self.assertIn(
            "@binding(2) var<storage, read_write> output: array<f32>;", shader
        )
        self.assertIn("@binding(3) var<uniform> params: Params;", shader)
        self.assertIn("let word = mask_values[index >> 2u];", shader)
        self.assertIn("((index & 3u) * 8u)", shader)
        self.assertIn("& 0xffu", shader)
        self.assertEqual(
            shader.count("output[params.output_offset + linear_index] ="), 1
        )
        self.assertNotIn("atomic", shader)
        self.assertNotRegex(shader, r"mask_values\s*\[[^]]+\]\s*=")
        self.assertIn("if (linear_index >= params.length) { return; }", shader)

    def test_functional_registration_is_narrow_and_fails_closed(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            'module.impl("masked_fill.Scalar", TORCH_FN(masked_fill_scalar));',
            source,
        )
        self.assertNotIn('module.impl("masked_fill_.Scalar"', source)
        self.assertNotIn('module.impl("masked_fill.Scalar_out"', source)
        self.assertIn("check_strided(self, operation, at::kFloat);", source)
        self.assertIn("check_strided(mask, operation, at::kBool);", source)
        self.assertIn("self.device() == mask.device()", source)
        self.assertIn("tensor.dim() <= kMaximumDimensions", source)
        self.assertIn("at::infer_size(self.sizes(), mask.sizes())", source)
        self.assertIn("broadcast result has more than eight dimensions", source)
        self.assertIn("tensor.stride(dim) >= 0", source)
        self.assertIn("view exceeds its GPUBuffer storage", source)
        self.assertIn("value.isIntegral(true) || value.isFloatingPoint()", source)
        self.assertIn("return value.toFloat();", source)
        self.assertNotIn("std::isfinite", source)
        self.assertNotIn("at::kCPU", source)
        self.assertNotIn(".cpu()", source)
        empty_return = source.index("if (output.numel() == 0)")
        first_binding = source.index("tensor_entry(0, self)")
        self.assertLess(empty_return, first_binding)

    def test_raw_gate_covers_broadcast_offsets_tails_and_scalar_values(self) -> None:
        gate = RAW_GATE.read_text(encoding="utf-8")
        self.assertIn('process.env.WEBGPU_ADAPTER ?? "swiftshader"', gate)
        self.assertIn('new Set(["hardware", "swiftshader"])', gate)
        self.assertIn("[2, 3, 5, 7]", gate)
        self.assertIn("[2, 1, 2, 1, 2, 1, 2, 5]", gate)
        self.assertIn("fillValue: -Infinity", gate)
        self.assertIn("fillValue: Infinity", gate)
        self.assertIn("fillValue: -3.25", gate)
        self.assertIn("fillValue: NaN", gate)
        self.assertIn("linear % 4 === 0 ? 255", gate)
        self.assertIn("device.pushErrorScope(\"validation\")", gate)
        self.assertIn("device.popErrorScope()", gate)

    def test_mul_tensor_dtype_router_remains_intact(self) -> None:
        source = MASKING_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "lhs.scalar_type() == at::kFloat && rhs.scalar_type() == at::kBool",
            source,
        )
        self.assertIn("return mul_bool_inplace(lhs, rhs);", source)
        self.assertIn("return at::_ops::mul_out::call(lhs, rhs, lhs);", source)

    def test_staging_patch_and_workflow_integrate_the_kernel(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(
            stage_webgpu_sources.SHADERS["masked_fill_scalar.wgsl"],
            "kMaskedFillScalar",
        )
        self.assertIn("llm_kernels/masked_fill.cpp", patch)
        self.assertIn("node tests/webgpu-masked-fill-wgsl.mjs", workflow)

        with tempfile.TemporaryDirectory() as temporary:
            pytorch = Path(temporary)
            (pytorch / "torch").mkdir()
            (pytorch / "torch" / "CMakeLists.txt").write_text(
                "# staging fixture\n", encoding="utf-8"
            )
            original_argv = sys.argv
            try:
                sys.argv = ["stage_webgpu_sources.py", str(pytorch)]
                self.assertEqual(stage_webgpu_sources.main(), 0)
            finally:
                sys.argv = original_argv
            staged = (
                pytorch
                / "third_party"
                / "pyodide-pytorch-webgpu"
                / "llm_kernels"
            )
            self.assertEqual(
                (staged / "masked_fill.cpp").read_bytes(), SOURCE.read_bytes()
            )
            self.assertEqual(
                (staged / "masked_fill_scalar.wgsl").read_bytes(),
                SHADER.read_bytes(),
            )
            embedded = (staged / "embedded_shaders.h").read_text(
                encoding="utf-8"
            )
            self.assertIn("inline constexpr char kMaskedFillScalar[]", embedded)

    def test_pinned_architecture_fixture_requires_the_functional_schema(self) -> None:
        fixture = FIXTURE.read_text(encoding="utf-8")
        self.assertEqual(fixture.count('"aten::masked_fill.Scalar"'), 2)
        self.assertIn('"name": "bloom"', fixture)
        self.assertIn('"name": "t5"', fixture)


if __name__ == "__main__":
    unittest.main()
