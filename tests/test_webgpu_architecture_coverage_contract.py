from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage_webgpu_sources  # noqa: E402


KERNELS = ROOT / "webgpu" / "llm_kernels"
PATCH = ROOT / "patches" / "pytorch" / (
    "0010-add-experimental-browser-webgpu-backend.patch"
)


class WebGPUArchitectureCoverageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.long_cpp = (KERNELS / "long_arithmetic.cpp").read_text(
            encoding="utf-8"
        )
        cls.long_shader = (KERNELS / "long_arithmetic.wgsl").read_text(
            encoding="utf-8"
        )
        cls.pow_shader = (KERNELS / "mixed_pow.wgsl").read_text(
            encoding="utf-8"
        )
        cls.matmul = (KERNELS / "matmul.cpp").read_text(encoding="utf-8")
        cls.baddbmm = (KERNELS / "baddbmm.wgsl").read_text(encoding="utf-8")
        cls.unary = (KERNELS / "browser_unary.cpp").read_text(encoding="utf-8")
        cls.patch = PATCH.read_text(encoding="utf-8")
        cls.transformers = (ROOT / "tests" / "transformers-webgpu.html").read_text(
            encoding="utf-8"
        )
        cls.gemma2 = (
            ROOT / "tests" / "transformers-gemma2-webgpu.html"
        ).read_text(encoding="utf-8")
        cls.raw_gate = (
            ROOT / "tests" / "webgpu-architecture-kernels-wgsl.mjs"
        ).read_text(encoding="utf-8")
        cls.workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )

    def test_restricted_long_arithmetic_preserves_profile(self) -> None:
        self.assertIn("static_assert(sizeof(ArithmeticParams) == 208);", self.long_cpp)
        self.assertIn("multiply_canonical", self.long_shader)
        self.assertIn("unsigned_multiply32", self.long_shader)
        self.assertIn("dim + params.lhs_ndim >= params.ndim", self.long_shader)
        self.assertIn("dim + params.rhs_ndim >= params.ndim", self.long_shader)
        self.assertIn("if (!canonical(result)) { result = invalid(); }", self.long_shader)
        self.assertIn("scalar is outside the signed-int32 WebGPU profile", self.long_cpp)
        for schema in (
            "add_.Tensor",
            "sub_.Tensor",
            "minimum",
            "minimum.out",
            "rsub.Scalar",
            "pow.Tensor_Tensor",
            "pow.Tensor_Tensor_out",
        ):
            self.assertIn(f'module.impl("{schema}"', self.long_cpp)
        self.assertNotIn("at::kCPU", self.long_cpp)
        self.assertNotIn(".cpu()", self.long_cpp)

    def test_mixed_power_checks_long_high_word(self) -> None:
        self.assertIn("static_assert(sizeof(PowParams) == 192);", self.long_cpp)
        self.assertIn("params.exponent_words == 2u", self.pow_shader)
        self.assertIn("expected_high", self.pow_shader)
        self.assertIn(
            "exponent[exponent_word + 1u] | 0x7fc00000u", self.pow_shader
        )
        self.assertIn("dim + params.exponent_ndim >= params.ndim", self.pow_shader)

    def test_baddbmm_is_fused_broadcasted_and_registered(self) -> None:
        self.assertIn("static_assert(sizeof(BaddbmmParams) == 112);", self.matmul)
        self.assertIn('module.impl("baddbmm", TORCH_FN(baddbmm_impl));', self.matmul)
        self.assertIn('module.impl("baddbmm_",', self.matmul)
        self.assertIn('module.impl("baddbmm.out",', self.matmul)
        self.assertIn("bitcast<f32>(params.beta_bits)", self.baddbmm)
        self.assertIn("bitcast<f32>(params.alpha_bits)", self.baddbmm)
        self.assertIn("if (beta != 0.0)", self.baddbmm)
        self.assertIn("params.self_sizes[input_dim] != 1u", self.baddbmm)
        self.assertNotIn("at::kCPU", self.matmul)

    def test_project_unary_router_preserves_float_and_adds_long(self) -> None:
        for schema in ("cos", "sin", "tanh", "exp", "abs", "rsqrt", "neg", "log"):
            self.assertIn(f'module.impl("{schema}"', self.unary)
            self.assertIn(f'module.impl("{schema}.out"', self.unary)
        self.assertIn("return abs_long_tensor(input);", self.unary)
        self.assertIn("return neg_long_tensor(input);", self.unary)
        self.assertIn("browser_unary.cpp", self.patch)
        self.assertNotIn("torch-webgpu/csrc/ops/trig.cpp", self.patch)

    def test_cpu_scalar_tensor_promotion_stays_on_webgpu(self) -> None:
        self.assertIn("cpu_scalar_value", self.patch)
        for expression in (
            "return at::add(lhs, *scalar, alpha);",
            "return at::mul(lhs, *scalar);",
            "return at::sub(lhs, *scalar, alpha);",
            "return at::div(lhs, *scalar);",
        ):
            self.assertIn(expression, self.patch)

    def test_staging_and_raw_numerical_gate_cover_new_shaders(self) -> None:
        self.assertEqual(
            stage_webgpu_sources.SHADERS["long_arithmetic.wgsl"],
            "kLongArithmetic",
        )
        self.assertEqual(stage_webgpu_sources.SHADERS["mixed_pow.wgsl"], "kMixedPow")
        self.assertEqual(stage_webgpu_sources.SHADERS["baddbmm.wgsl"], "kBaddbmm")
        for filename in ("long_arithmetic.wgsl", "mixed_pow.wgsl", "baddbmm.wgsl"):
            self.assertIn(filename, self.raw_gate)
        for case in (
            "add.Tensor alpha",
            "mul.Tensor",
            "minimum",
            "rsub.Scalar",
            "pow.Tensor_Tensor",
            "baddbmm",
            "baddbmm beta zero ignores NaN self",
        ):
            self.assertIn(case, self.raw_gate)
        self.assertIn("module.getCompilationInfo", self.raw_gate)
        self.assertIn("device.popErrorScope()", self.raw_gate)
        self.assertIn("node tests/webgpu-architecture-kernels-wgsl.mjs", self.workflow)

    def test_ten_family_browser_profile_is_mandatory(self) -> None:
        self.assertIn('if spec["name"] in ("bloom", "t5")', self.transformers)
        for name in (
            "qwen2",
            "llama",
            "mistral",
            "gpt2",
            "bert",
            "phi3",
            "opt",
            "bloom",
            "t5",
        ):
            self.assertIn(f'"{name}"', self.transformers)
        self.assertIn("actual = getattr(gpu_outputs, spec[\"output\"]).cpu()", self.transformers)
        self.assertNotIn("await torch.webgpu.to_cpu_async", self.transformers)
        self.assertIn("actual = actual_gpu.cpu()", self.gemma2)
        self.assertNotIn("await torch.webgpu.to_cpu_async", self.gemma2)
        self.assertIn('entry["cpu_fallbacks"] == 0', self.transformers)

    def test_jspi_api_removes_default_python_await_boundaries(self) -> None:
        for definition in (
            "async def init_async",
            "def init(",
            "async def synchronize_async",
            "def synchronize(",
            "def to_cpu_sync(",
        ):
            self.assertIn(definition, self.patch)
        self.assertIn("from pyodide.ffi import can_run_sync, run_sync", self.patch)
        self.assertIn("runPythonAsync() or callPromising()", self.patch)
        self.assertIn('"init_async",', self.patch)
        self.assertIn('"synchronize_async",', self.patch)


if __name__ == "__main__":
    unittest.main()
