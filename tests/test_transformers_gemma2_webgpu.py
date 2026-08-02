from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "site"))

import transformers_gemma2_webgpu as gemma2  # noqa: E402


class FakeTensor:
    def __init__(
        self,
        device: str,
        dtype: object,
        shape: tuple[int, ...],
        *,
        contiguous: bool = True,
        requires_grad: bool = False,
    ) -> None:
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)
        self.requires_grad = requires_grad
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


class Gemma2RmsNormAdapterTests(unittest.TestCase):
    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        include_class: bool = True,
        include_op: bool = True,
        bad_signature: bool = False,
    ) -> tuple[dict[str, ModuleType], ModuleType, list[tuple[object, ...]]]:
        fused_calls: list[tuple[object, ...]] = []
        float32 = object()

        def fused(*args: object) -> tuple[object, ...]:
            fused_calls.append(args)
            return ("fused", *args)

        torch = ModuleType("torch")
        torch.float32 = float32  # type: ignore[attr-defined]
        torch.is_grad_enabled = lambda: False  # type: ignore[attr-defined]
        torch.ops = SimpleNamespace(  # type: ignore[attr-defined]
            webgpu=SimpleNamespace(gemma_rms_norm=fused if include_op else None)
        )
        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        model_module = ModuleType(gemma2._GEMMA2_MODULE)
        if include_class:
            namespace: dict[str, object] = {"__name__": gemma2._GEMMA2_MODULE}
            parameters = "self, x, extra=None" if bad_signature else "self, x"
            exec(
                "\n".join(
                    (
                        "class Gemma2RMSNorm:",
                        f"    def forward({parameters}):",
                        "        self.original_calls.append(x)",
                        "        return ('upstream', x)",
                    )
                ),
                namespace,
            )
            model_module.Gemma2RMSNorm = namespace[  # type: ignore[attr-defined]
                "Gemma2RMSNorm"
            ]
        modules = {
            "torch": torch,
            "transformers": transformers,
            gemma2._GEMMA2_MODULE: model_module,
        }
        return modules, torch, fused_calls

    def make_instance(
        self,
        modules: dict[str, ModuleType],
        torch: ModuleType,
        *,
        device: str = "webgpu",
        dtype: object | None = None,
        contiguous_weight: bool = True,
        requires_grad: bool = False,
    ) -> tuple[object, FakeTensor]:
        target_class = modules[gemma2._GEMMA2_MODULE].Gemma2RMSNorm
        instance = target_class()
        instance.original_calls = []
        instance.eps = 1e-6
        selected_dtype = torch.float32 if dtype is None else dtype
        instance.weight = FakeTensor(
            device,
            selected_dtype,
            (16,),
            contiguous=contiguous_weight,
            requires_grad=requires_grad,
        )
        value = FakeTensor(
            device,
            selected_dtype,
            (2, 4, 16),
            requires_grad=requires_grad,
        )
        return instance, value

    def test_exact_webgpu_call_uses_offset_weight_operator(self) -> None:
        modules, torch, fused_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            diagnostics = gemma2.enable_webgpu_gemma2_rms_norm()
            instance, value = self.make_instance(modules, torch)
            result = instance.forward(value)

        self.assertEqual(result[0], "fused")
        self.assertEqual(fused_calls, [(value, instance.weight, 1e-6)])
        self.assertEqual(instance.original_calls, [])
        self.assertEqual(diagnostics["dispatches_saved_per_call"], 5)
        self.assertEqual(diagnostics["two_layer_fixture_dispatches_saved"], 45)

    def test_cpu_and_unsupported_calls_preserve_upstream(self) -> None:
        modules, torch, fused_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            gemma2.enable_webgpu_gemma2_rms_norm()
            cases = [
                self.make_instance(modules, torch, device="cpu"),
                self.make_instance(modules, torch, dtype=object()),
                self.make_instance(
                    modules, torch, contiguous_weight=False
                ),
            ]
            for instance, value in cases:
                self.assertEqual(instance.forward(value), ("upstream", value))
                self.assertEqual(instance.original_calls, [value])

            torch.is_grad_enabled = lambda: True  # type: ignore[attr-defined]
            instance, value = self.make_instance(
                modules, torch, requires_grad=True
            )
            self.assertEqual(instance.forward(value), ("upstream", value))

        self.assertEqual(fused_calls, [])

    def test_enable_is_idempotent(self) -> None:
        modules, _, _ = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            first = gemma2.enable_webgpu_gemma2_rms_norm()
            forward = modules[gemma2._GEMMA2_MODULE].Gemma2RMSNorm.forward
            second = gemma2.enable_webgpu_gemma2_rms_norm()

        self.assertEqual(first["newly_patched"], 1)
        self.assertEqual(second["newly_patched"], 0)
        self.assertEqual(second["already_patched"], 1)
        self.assertIs(
            modules[gemma2._GEMMA2_MODULE].Gemma2RMSNorm.forward,
            forward,
        )

    def test_version_class_signature_and_operator_fail_closed(self) -> None:
        cases = (
            self.make_modules(transformers_version="4.47.0")[0],
            self.make_modules(include_class=False)[0],
            self.make_modules(include_op=False)[0],
            self.make_modules(bad_signature=True)[0],
        )
        for modules in cases:
            model_module = modules[gemma2._GEMMA2_MODULE]
            target_class = getattr(model_module, "Gemma2RMSNorm", None)
            original = getattr(target_class, "forward", None)
            with self.subTest(modules=modules), mock.patch.dict(
                sys.modules, modules
            ):
                with self.assertRaises(gemma2.Gemma2WebGPUProfileError):
                    gemma2.enable_webgpu_gemma2_rms_norm()
            if target_class is not None:
                self.assertIs(target_class.forward, original)


class Gemma2ScalarNormalizerAdapterTests(unittest.TestCase):
    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        include_full: bool = True,
        bad_signature: bool = False,
    ) -> tuple[
        dict[str, ModuleType],
        ModuleType,
        list[tuple[object, ...]],
        list[tuple[object, ...]],
    ]:
        tensor_calls: list[tuple[object, ...]] = []
        full_calls: list[tuple[object, ...]] = []
        float32 = object()
        torch = ModuleType("torch")
        torch.float32 = float32  # type: ignore[attr-defined]

        def tensor(value: object, **kwargs: object) -> tuple[object, ...]:
            call = (value, kwargs)
            tensor_calls.append(call)
            return ("cpu-tensor", *call)

        def full(
            shape: tuple[object, ...],
            value: object,
            *,
            dtype: object,
            device: object,
        ) -> FakeTensor:
            full_calls.append((shape, value, dtype, device))
            return FakeTensor("webgpu", dtype, ())

        torch.tensor = tensor  # type: ignore[attr-defined]
        if include_full:
            torch.full = full  # type: ignore[attr-defined]
        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        model_module = ModuleType(gemma2._GEMMA2_MODULE)
        model_module.torch = torch  # type: ignore[attr-defined]
        parameters = (
            "self, input_ids=None, attention_mask=None, position_ids=None, "
            "past_key_values=None, inputs_embeds=None, use_cache=None, "
            "output_attentions=None, output_hidden_states=None, "
            "return_dict=None, cache_position=None"
        )
        if bad_signature:
            parameters += ", unexpected=None"
        exec(
            "\n".join(
                (
                    "class Gemma2Model:",
                    f"    def forward({parameters}):",
                    "        selected = inputs_embeds if inputs_embeds is not None else input_ids",
                    "        normalizer = torch.tensor(",
                    "            self.config.hidden_size**0.5,",
                    "            dtype=self.embed_tokens.weight.dtype,",
                    "        )",
                    "        self.normalizers.append(normalizer)",
                    "        if self.raise_after_normalizer:",
                    "            raise RuntimeError('forward failure')",
                    "        return ('upstream', selected, normalizer)",
                )
            ),
            model_module.__dict__,
        )
        modules = {
            "torch": torch,
            "transformers": transformers,
            gemma2._GEMMA2_MODULE: model_module,
        }
        return modules, torch, tensor_calls, full_calls

    def make_instance(
        self,
        modules: dict[str, ModuleType],
        torch: ModuleType,
    ) -> object:
        instance = modules[gemma2._GEMMA2_MODULE].Gemma2Model()
        instance.config = SimpleNamespace(hidden_size=16)
        instance.embed_tokens = SimpleNamespace(
            weight=FakeTensor("webgpu", torch.float32, (64, 16))
        )
        instance.normalizers = []
        instance.raise_after_normalizer = False
        return instance

    def test_webgpu_forward_substitutes_and_caches_device_scalar(self) -> None:
        modules, torch, tensor_calls, full_calls = self.make_modules()
        model_module = modules[gemma2._GEMMA2_MODULE]
        with mock.patch.dict(sys.modules, modules):
            diagnostics = gemma2.enable_webgpu_gemma2_scalar_normalizer()
            instance = self.make_instance(modules, torch)
            input_ids = FakeTensor("webgpu", object(), (1, 4))
            first = instance.forward(input_ids=input_ids)
            second = instance.forward(input_ids=input_ids)

        self.assertEqual(diagnostics["mixed_device_mul_avoided_per_forward"], 1)
        self.assertEqual(tensor_calls, [])
        self.assertEqual(len(full_calls), 1)
        self.assertIs(first[2], second[2])
        self.assertEqual(first[2].device.type, "webgpu")
        self.assertIs(model_module.torch, torch)

    def test_cpu_forward_is_bit_for_bit_upstream(self) -> None:
        modules, torch, tensor_calls, full_calls = self.make_modules()
        model_module = modules[gemma2._GEMMA2_MODULE]
        with mock.patch.dict(sys.modules, modules):
            gemma2.enable_webgpu_gemma2_scalar_normalizer()
            instance = self.make_instance(modules, torch)
            input_ids = FakeTensor("cpu", object(), (1, 4))
            result = instance.forward(input_ids=input_ids)

        self.assertEqual(result[2][0], "cpu-tensor")
        self.assertEqual(len(tensor_calls), 1)
        self.assertEqual(full_calls, [])
        self.assertIs(model_module.torch, torch)

    def test_exception_restores_module_global(self) -> None:
        modules, torch, _, _ = self.make_modules()
        model_module = modules[gemma2._GEMMA2_MODULE]
        with mock.patch.dict(sys.modules, modules):
            gemma2.enable_webgpu_gemma2_scalar_normalizer()
            instance = self.make_instance(modules, torch)
            instance.raise_after_normalizer = True
            with self.assertRaisesRegex(RuntimeError, "forward failure"):
                instance.forward(
                    input_ids=FakeTensor("webgpu", object(), (1, 4))
                )
        self.assertIs(model_module.torch, torch)

    def test_enable_is_idempotent(self) -> None:
        modules, _, _, _ = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            first = gemma2.enable_webgpu_gemma2_scalar_normalizer()
            forward = modules[gemma2._GEMMA2_MODULE].Gemma2Model.forward
            second = gemma2.enable_webgpu_gemma2_scalar_normalizer()
        self.assertEqual(first["newly_patched"], 1)
        self.assertEqual(second["already_patched"], 1)
        self.assertIs(
            modules[gemma2._GEMMA2_MODULE].Gemma2Model.forward,
            forward,
        )

    def test_version_signature_full_and_global_binding_fail_closed(self) -> None:
        cases = [
            self.make_modules(transformers_version="4.47.0"),
            self.make_modules(include_full=False),
            self.make_modules(bad_signature=True),
        ]
        changed_global = self.make_modules()
        changed_global[0][gemma2._GEMMA2_MODULE].torch = object()
        cases.append(changed_global)
        for modules, _torch, _tensor_calls, _full_calls in cases:
            target_class = modules[gemma2._GEMMA2_MODULE].Gemma2Model
            original = target_class.forward
            with self.subTest(modules=modules), mock.patch.dict(
                sys.modules, modules
            ):
                with self.assertRaises(gemma2.Gemma2WebGPUProfileError):
                    gemma2.enable_webgpu_gemma2_scalar_normalizer()
            self.assertIs(target_class.forward, original)


class Gemma2KernelSourceContractTests(unittest.TestCase):
    def test_mask_kernels_own_packed_bool_words(self) -> None:
        gt_shader = (ROOT / "webgpu/llm_kernels/gt_tensor.wgsl").read_text()
        triangular_shader = (
            ROOT / "webgpu/llm_kernels/triangular.wgsl"
        ).read_text()
        self.assertIn("let first_index = output_word * 4u", gt_shader)
        self.assertIn("output[output_word] = packed", gt_shader)
        self.assertIn("let first_index = unit * 4u", triangular_shader)
        self.assertIn("output[unit] = packed", triangular_shader)

    def test_exact_gemma_mask_operator_set_is_registered(self) -> None:
        source = (ROOT / "webgpu/llm_kernels/masking.cpp").read_text()
        for operator in (
            'module.impl("gt.Tensor"',
            'module.impl("triu"',
            'module.impl("tril"',
            'module.impl("where.self"',
            'module.impl("mul_.Tensor"',
        ):
            self.assertIn(operator, source)
        self.assertIn("signed-int32-valued", source)
        self.assertIn("requires a contiguous left-hand tensor", source)

    def test_bool_mul_specialization_preserves_general_inplace_mul(self) -> None:
        source = (ROOT / "webgpu/llm_kernels/masking.cpp").read_text()
        self.assertIn(
            'module.impl("mul_.Tensor", TORCH_FN(mul_tensor_inplace))',
            source,
        )
        self.assertNotIn(
            'module.impl("mul_.Tensor", TORCH_FN(mul_bool_inplace))',
            source,
        )
        self.assertIn(
            "lhs.scalar_type() == at::kFloat && "
            "rhs.scalar_type() == at::kBool",
            source,
        )
        self.assertIn("at::_ops::mul_out::call(lhs, rhs, lhs)", source)

    def test_gemma_norm_uses_offset_weight_semantics(self) -> None:
        shader = (
            ROOT / "webgpu/llm_kernels/gemma_rms_norm.wgsl"
        ).read_text()
        self.assertIn("(1.0 + offset_weight", shader)
        self.assertNotIn("value *= offset_weight", shader)


if __name__ == "__main__":
    unittest.main()
