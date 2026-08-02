from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / "webgpu" / "llm_kernels" / "linear_gemv_q8_s4.wgsl"
SOURCE = ROOT / "webgpu" / "llm_kernels" / "q8_linear.cpp"
HELPER = ROOT / "site" / "transformers_q8.py"
BUILD_WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"
WEBGPU_PAGE = ROOT / "tests" / "webgpu.html"


def struct_fields(source: str, name: str) -> list[str]:
    match = re.search(rf"struct {name} \{{(.*?)\n\}};", source, re.DOTALL)
    if match is None:
        raise AssertionError(f"missing struct {name}")
    fields: list[str] = []
    for declaration in match.group(1).splitlines():
        declaration = declaration.strip()
        if not declaration or declaration.startswith("//"):
            continue
        if declaration.startswith("std::uint32_t "):
            field = declaration.removeprefix("std::uint32_t ").removesuffix(";")
        else:
            field = declaration.split(":", 1)[0].strip()
        if field.startswith("padding") or field.startswith("_pad"):
            continue
        fields.append(field)
    return fields


class Q8LinearContractTests(unittest.TestCase):
    def test_shader_uses_the_versioned_group128_q8_layout(self) -> None:
        shader = SHADER.read_text(encoding="utf-8")
        self.assertIn("enable subgroups;", shader)
        self.assertIn("const SUBGROUP_SIZE: u32 = 32u;", shader)
        self.assertIn("const GROUP_SIZE: u32 = 128u;", shader)
        self.assertIn("const FORMAT_VERSION: u32 = 1u;", shader)
        self.assertIn("unpack4x8snorm(word)", shader)
        self.assertIn("subgroupAdd(accumulator)", shader)
        self.assertIn("subgroupBroadcast(row_scales, 0u)", shader)
        self.assertIn("if (!valid_layout())", shader)
        for binding in range(6):
            self.assertIn(f"@binding({binding})", shader)

    def test_cpp_and_wgsl_uniform_abis_match(self) -> None:
        shader = SHADER.read_text(encoding="utf-8")
        source = SOURCE.read_text(encoding="utf-8")
        expected = [
            "columns",
            "inner",
            "input_offset",
            "packed_weight_offset",
            "scale_offset",
            "bias_offset",
            "output_offset",
            "has_bias",
            "words_per_row",
            "groups_per_row",
            "group_size",
            "format_version",
        ]
        self.assertEqual(struct_fields(shader, "Params"), expected)
        self.assertEqual(struct_fields(source, "Q8LinearParams"), expected)
        self.assertIn("static_assert(sizeof(Q8LinearParams) == 64);", source)

    def test_custom_op_fails_closed_at_every_unsafe_boundary(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("q8_linear(Tensor input, Tensor packed_weight", source)
        self.assertIn("check_contiguous_tensor(packed_weight, at::kInt", source)
        self.assertIn("check_contiguous_tensor(scales, at::kFloat", source)
        self.assertIn("rows == 1", source)
        self.assertIn("inner % kGroupSize == 0", source)
        self.assertIn("packed_weight.size(1) == inner / kValuesPerWord", source)
        self.assertIn("scales.size(1) == inner / kGroupSize", source)
        self.assertIn("q8_linear_subgroup_s4_supported()", source)
        self.assertIn("workgroups <= kMaxWorkgroupsPerDimension", source)
        self.assertIn("tensor exceeds its GPUBuffer storage", source)
        self.assertNotIn("CPU", source.split("q8_linear_impl", 1)[1])

    def test_python_conversion_is_opt_in_and_drops_float_weights(self) -> None:
        helper = HELPER.read_text(encoding="utf-8")
        ast.parse(helper, filename=str(HELPER))
        self.assertIn("class WebGPUQ8Linear(nn.Module):", helper)
        self.assertIn("def convert_linear_modules_q8_(", helper)
        self.assertIn("Q8 conversion preflight failed", helper)
        self.assertIn("parent._modules[child_name] = replacement", helper)
        self.assertIn("packed_cpu.to(device)", helper)
        self.assertNotIn("register_parameter(\"weight\"", helper)
        self.assertNotIn("self.weight =", helper)
        self.assertIn("expected exactly one", helper)
        self.assertIn("torch.ops.webgpu.q8_linear", helper)

    def test_general_transformers_bootstrap_does_not_enable_q8_implicitly(self) -> None:
        bootstrap = (ROOT / "site" / "transformers_browser_bootstrap.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("convert_linear_modules_q8_", bootstrap)
        self.assertNotIn("WebGPUQ8Linear", bootstrap)

    def test_build_requires_the_raw_q8_browser_gate(self) -> None:
        workflow = BUILD_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("node tests/webgpu-packed-gemv-wgsl.mjs", workflow)

    def test_core_page_enables_and_reports_fixed32_subgroups(self) -> None:
        page = WEBGPU_PAGE.read_text(encoding="utf-8")
        self.assertIn("const adapterInfo = adapter.info ?? {};", page)
        self.assertIn('adapter.features.has("subgroups")', page)
        self.assertIn("adapterInfo.subgroupMinSize === 32", page)
        self.assertIn("adapterInfo.subgroupMaxSize === 32", page)
        self.assertIn(
            'const requiredFeatures = fixed32Subgroups ? ["subgroups"] : [];',
            page,
        )
        self.assertIn(
            "const device = await adapter.requestDevice({ requiredFeatures });",
            page,
        )
        self.assertIn('subgroups: device.features.has("subgroups"),', page)
        self.assertIn(
            "subgroupMinSize: adapterInfo.subgroupMinSize ?? null,", page
        )
        self.assertIn(
            "subgroupMaxSize: adapterInfo.subgroupMaxSize ?? null,", page
        )


if __name__ == "__main__":
    unittest.main()
