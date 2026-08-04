from __future__ import annotations

import sys
import unittest
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "site"))

import transformers_browser_bootstrap as bootstrap  # noqa: E402


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
        self._contiguous = contiguous
        self.requires_grad = requires_grad

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result


class RmsNormFusionTests(unittest.TestCase):
    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        omit_class: str | None = None,
    ) -> tuple[dict[str, ModuleType], ModuleType, list[tuple[object, ...]]]:
        fused_calls: list[tuple[object, ...]] = []
        float32 = object()

        def rms_norm(
            hidden_states: object,
            normalized_shape: tuple[int, ...],
            weight: object,
            eps: float,
        ) -> tuple[object, ...]:
            call = (hidden_states, normalized_shape, weight, eps)
            fused_calls.append(call)
            return ("fused", *call)

        torch = ModuleType("torch")
        torch.float32 = float32  # type: ignore[attr-defined]
        torch.nn = SimpleNamespace(  # type: ignore[attr-defined]
            functional=SimpleNamespace(rms_norm=rms_norm)
        )
        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        modules = {"torch": torch, "transformers": transformers}

        for module_name, class_name in bootstrap._RMS_NORM_TARGETS:
            module = ModuleType(module_name)
            if class_name != omit_class:
                namespace: dict[str, object] = {"__name__": module_name}
                exec(
                    "\n".join(
                        (
                            f"class {class_name}:",
                            "    def forward(self, hidden_states):",
                            "        self.original_calls.append(hidden_states)",
                            "        return ('upstream', hidden_states)",
                        )
                    ),
                    namespace,
                )
                setattr(module, class_name, namespace[class_name])
            modules[module_name] = module
        return modules, torch, fused_calls

    def test_exact_pinned_classes_fuse_only_float32_webgpu_calls(self) -> None:
        modules, torch, fused_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            diagnostics = bootstrap.enable_webgpu_rms_norm_fusion()

            self.assertTrue(diagnostics["enabled"])
            self.assertEqual(diagnostics["transformers_version"], "4.46.3")
            self.assertEqual(diagnostics["newly_patched"], 4)
            self.assertEqual(diagnostics["already_patched"], 0)
            self.assertEqual(
                [target["status"] for target in diagnostics["targets"]],
                ["patched", "patched", "patched", "patched"],
            )

            for module_name, class_name in bootstrap._RMS_NORM_TARGETS:
                target_class = getattr(modules[module_name], class_name)
                instance = target_class()
                instance.original_calls = []
                instance.variance_epsilon = 1e-6
                instance.weight = FakeTensor("webgpu", torch.float32, (16,))

                cpu_input = FakeTensor("cpu", torch.float32, (2, 4, 16))
                self.assertEqual(
                    instance.forward(cpu_input), ("upstream", cpu_input)
                )
                self.assertEqual(instance.original_calls, [cpu_input])

                gpu_input = FakeTensor("webgpu", torch.float32, (2, 4, 16))
                fused = instance.forward(gpu_input)
                self.assertEqual(fused[0], "fused")
                self.assertIs(fused[1], gpu_input)
                self.assertEqual(fused[2], (16,))
                self.assertIs(fused[3], instance.weight)
                self.assertEqual(fused[4], 1e-6)
                self.assertEqual(instance.original_calls, [cpu_input])

                other_dtype = FakeTensor("webgpu", object(), (2, 4, 16))
                self.assertEqual(
                    instance.forward(other_dtype), ("upstream", other_dtype)
                )
                self.assertEqual(instance.original_calls, [cpu_input, other_dtype])

        self.assertEqual(len(fused_calls), 4)

    def test_enable_is_idempotent(self) -> None:
        modules, _, _ = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            first = bootstrap.enable_webgpu_rms_norm_fusion()
            forward_methods = [
                getattr(modules[module_name], class_name).forward
                for module_name, class_name in bootstrap._RMS_NORM_TARGETS
            ]
            second = bootstrap.enable_webgpu_rms_norm_fusion()

            self.assertEqual(first["newly_patched"], 4)
            self.assertEqual(second["newly_patched"], 0)
            self.assertEqual(second["already_patched"], 4)
            self.assertEqual(
                [target["status"] for target in second["targets"]],
                [
                    "already_enabled",
                    "already_enabled",
                    "already_enabled",
                    "already_enabled",
                ],
            )
            self.assertEqual(
                forward_methods,
                [
                    getattr(modules[module_name], class_name).forward
                    for module_name, class_name in bootstrap._RMS_NORM_TARGETS
                ],
            )

    def test_new_transformers_version_fails_closed(self) -> None:
        modules, _, _ = self.make_modules(transformers_version="4.47.0")
        originals = {
            class_name: getattr(modules[module_name], class_name).forward
            for module_name, class_name in bootstrap._RMS_NORM_TARGETS
        }
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                bootstrap.TransformersBrowserProfileError,
                r"supports exactly Transformers 4\.46\.3; found 4\.47\.0",
            ):
                bootstrap.enable_webgpu_rms_norm_fusion()

        for module_name, class_name in bootstrap._RMS_NORM_TARGETS:
            self.assertIs(
                getattr(modules[module_name], class_name).forward,
                originals[class_name],
            )

    def test_missing_pinned_class_fails_atomically(self) -> None:
        modules, _, _ = self.make_modules(omit_class="MistralRMSNorm")
        originals = {
            class_name: getattr(modules[module_name], class_name).forward
            for module_name, class_name in bootstrap._RMS_NORM_TARGETS
            if hasattr(modules[module_name], class_name)
        }
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(
                bootstrap.TransformersBrowserProfileError,
                r"expected class .*MistralRMSNorm; no classes were changed",
            ):
                bootstrap.enable_webgpu_rms_norm_fusion()

        for module_name, class_name in bootstrap._RMS_NORM_TARGETS:
            if class_name in originals:
                self.assertIs(
                    getattr(modules[module_name], class_name).forward,
                    originals[class_name],
                )


class SwiGluFusionTests(unittest.TestCase):
    class FakeSiLU:
        def __init__(self, *, inplace: bool = False) -> None:
            self.inplace = inplace
            self._forward_pre_hooks: dict[object, object] = {}
            self._forward_hooks: dict[object, object] = {}
            self._backward_pre_hooks: dict[object, object] = {}
            self._backward_hooks: dict[object, object] = {}

    class Projection:
        def __init__(self, weight: FakeTensor, bias: FakeTensor | None) -> None:
            self.weight = weight
            self.bias = bias
            self._forward_pre_hooks: dict[object, object] = {}
            self._forward_hooks: dict[object, object] = {}
            self._backward_pre_hooks: dict[object, object] = {}
            self._backward_hooks: dict[object, object] = {}

    class DownProjection:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def __call__(self, value: object) -> tuple[str, object]:
            self.calls.append(value)
            return ("down", value)

    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        omit_class: str | None = None,
        include_op: bool = True,
        include_linear: bool = True,
    ) -> tuple[dict[str, ModuleType], ModuleType, list[tuple[object, ...]]]:
        fused_calls: list[tuple[object, ...]] = []
        float32 = object()

        def fused_swiglu(*args: object) -> tuple[str, object]:
            fused_calls.append(args)
            return ("fused-intermediate", args[0])

        global_hooks = SimpleNamespace(
            _global_forward_pre_hooks={},
            _global_forward_hooks={},
            _global_backward_pre_hooks={},
            _global_backward_hooks={},
        )
        torch = ModuleType("torch")
        torch.float32 = float32  # type: ignore[attr-defined]
        torch.is_grad_enabled = lambda: False  # type: ignore[attr-defined]
        torch.nn = SimpleNamespace(  # type: ignore[attr-defined]
            SiLU=self.FakeSiLU,
            Linear=self.Projection if include_linear else None,
            modules=SimpleNamespace(module=global_hooks),
        )
        torch.ops = SimpleNamespace(  # type: ignore[attr-defined]
            webgpu=SimpleNamespace(
                fused_swiglu=fused_swiglu if include_op else None
            )
        )
        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        modules = {"torch": torch, "transformers": transformers}

        for module_name, class_name, input_name in bootstrap._SWIGLU_TARGETS:
            module = ModuleType(module_name)
            if class_name != omit_class:
                namespace: dict[str, object] = {"__name__": module_name}
                exec(
                    "\n".join(
                        (
                            f"class {class_name}:",
                            f"    def forward(self, {input_name}):",
                            f"        self.original_calls.append({input_name})",
                            f"        return ('upstream', {input_name})",
                        )
                    ),
                    namespace,
                )
                setattr(module, class_name, namespace[class_name])
            modules[module_name] = module
        return modules, torch, fused_calls

    def make_instance(
        self,
        modules: dict[str, ModuleType],
        torch: ModuleType,
        *,
        class_index: int = 0,
        input_shape: tuple[int, ...] = (1, 1, 8),
        contiguous: bool = True,
        dtype: object | None = None,
        device: str = "webgpu",
    ) -> tuple[object, FakeTensor]:
        module_name, class_name, _ = bootstrap._SWIGLU_TARGETS[class_index]
        instance = getattr(modules[module_name], class_name)()
        instance.original_calls = []
        weight_dtype = torch.float32 if dtype is None else dtype
        gate_weight = FakeTensor("webgpu", weight_dtype, (16, 8))
        up_weight = FakeTensor("webgpu", weight_dtype, (16, 8))
        bias = (
            FakeTensor("webgpu", torch.float32, (16,))
            if class_name == "LlamaMLP"
            else None
        )
        instance.gate_proj = self.Projection(gate_weight, bias)
        instance.up_proj = self.Projection(up_weight, bias)
        instance.down_proj = self.DownProjection()
        instance.act_fn = self.FakeSiLU()
        instance.config = SimpleNamespace(pretraining_tp=1)
        hidden = FakeTensor(
            device,
            torch.float32 if dtype is None else dtype,
            input_shape,
            contiguous=contiguous,
        )
        return instance, hidden

    def test_exact_pinned_decode_calls_use_one_fused_dispatch(self) -> None:
        modules, torch, fused_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            diagnostics = bootstrap.enable_webgpu_swiglu_fusion()

            self.assertEqual(diagnostics["newly_patched"], 3)
            self.assertEqual(diagnostics["upstream_dispatches_before_down_proj"], 4)
            self.assertEqual(diagnostics["fused_dispatches_before_down_proj"], 1)
            self.assertEqual(diagnostics["dispatches_saved_per_fused_call"], 3)
            for index, _target in enumerate(bootstrap._SWIGLU_TARGETS):
                instance, hidden = self.make_instance(
                    modules,
                    torch,
                    class_index=index,
                )
                result = instance.forward(hidden)
                self.assertEqual(result[0], "down")
                self.assertEqual(result[1], ("fused-intermediate", hidden))
                self.assertEqual(instance.original_calls, [])
                self.assertEqual(instance.down_proj.calls, [result[1]])

        self.assertEqual(len(fused_calls), 3)
        for call in fused_calls:
            self.assertEqual(len(call), 5)

    def test_cpu_and_unsupported_calls_execute_upstream_unchanged(self) -> None:
        modules, torch, fused_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            bootstrap.enable_webgpu_swiglu_fusion()

            cases: list[tuple[object, FakeTensor]] = []
            cases.append(self.make_instance(modules, torch, device="cpu"))
            cases.append(
                self.make_instance(modules, torch, input_shape=(1, 2, 8))
            )
            cases.append(self.make_instance(modules, torch, contiguous=False))
            cases.append(self.make_instance(modules, torch, dtype=object()))

            hooked, hooked_input = self.make_instance(modules, torch)
            hooked.gate_proj._forward_hooks[1] = object()
            cases.append((hooked, hooked_input))

            wrong_activation, activation_input = self.make_instance(
                modules, torch
            )
            wrong_activation.act_fn = object()
            cases.append((wrong_activation, activation_input))

            tensor_parallel, tp_input = self.make_instance(
                modules,
                torch,
                class_index=1,
            )
            tensor_parallel.config.pretraining_tp = 2
            cases.append((tensor_parallel, tp_input))

            custom_projection, custom_projection_input = self.make_instance(
                modules, torch
            )
            original_gate = custom_projection.gate_proj
            custom_projection.gate_proj = SimpleNamespace(
                weight=original_gate.weight,
                bias=original_gate.bias,
            )
            cases.append((custom_projection, custom_projection_input))

            for instance, hidden in cases:
                self.assertEqual(
                    instance.forward(hidden),
                    ("upstream", hidden),
                )
                self.assertEqual(instance.original_calls, [hidden])
                self.assertEqual(instance.down_proj.calls, [])

        self.assertEqual(fused_calls, [])

    def test_enable_is_idempotent(self) -> None:
        modules, _, _ = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            first = bootstrap.enable_webgpu_swiglu_fusion()
            forwards = [
                getattr(modules[module_name], class_name).forward
                for module_name, class_name, _ in bootstrap._SWIGLU_TARGETS
            ]
            second = bootstrap.enable_webgpu_swiglu_fusion()

            self.assertEqual(first["newly_patched"], 3)
            self.assertEqual(second["newly_patched"], 0)
            self.assertEqual(second["already_patched"], 3)
            self.assertEqual(
                forwards,
                [
                    getattr(modules[module_name], class_name).forward
                    for module_name, class_name, _ in bootstrap._SWIGLU_TARGETS
                ],
            )

    def test_version_class_and_operator_fail_without_partial_patch(self) -> None:
        module_sets = (
            self.make_modules(transformers_version="4.47.0")[0],
            self.make_modules(omit_class="MistralMLP")[0],
            self.make_modules(include_op=False)[0],
            self.make_modules(include_linear=False)[0],
        )
        for modules in module_sets:
            originals = {
                class_name: getattr(modules[module_name], class_name).forward
                for module_name, class_name, _ in bootstrap._SWIGLU_TARGETS
                if hasattr(modules[module_name], class_name)
            }
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaises(
                    bootstrap.TransformersBrowserProfileError
                ):
                    bootstrap.enable_webgpu_swiglu_fusion()
            for module_name, class_name, _ in bootstrap._SWIGLU_TARGETS:
                if class_name in originals:
                    self.assertIs(
                        getattr(modules[module_name], class_name).forward,
                        originals[class_name],
                    )


class BertSdpaMaskCompatibilityTests(unittest.TestCase):
    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        omit_prepare: bool = False,
        bad_prepare_signature: bool = False,
        bad_expand_signature: bool = False,
        include_operators: bool = True,
    ) -> tuple[dict[str, ModuleType], list[tuple[object, ...]]]:
        original_calls: list[tuple[object, ...]] = []
        torch = ModuleType("torch")
        if include_operators:
            torch.ops = SimpleNamespace(  # type: ignore[attr-defined]
                aten=SimpleNamespace(
                    sub=SimpleNamespace(Scalar=lambda tensor, scalar: tensor),
                    eq=SimpleNamespace(Scalar=lambda tensor, scalar: tensor),
                )
            )
            torch.neg = lambda tensor: tensor  # type: ignore[attr-defined]
            torch.finfo = lambda dtype: SimpleNamespace(min=-1.0)  # type: ignore[attr-defined]
        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        utils = ModuleType(bootstrap._BERT_SDPA_MASK_UTILS_MODULE)
        namespace: dict[str, object] = {
            "__name__": bootstrap._BERT_SDPA_MASK_UTILS_MODULE,
            "original_calls": original_calls,
        }
        prepare_parameters = (
            "mask, dtype" if bad_prepare_signature else "mask, dtype, tgt_len=None"
        )
        expand_parameters = (
            "mask, dtype" if bad_expand_signature else "mask, dtype, tgt_len=None"
        )
        exec(
            "\n".join(
                (
                    f"def {bootstrap._BERT_SDPA_MASK_HELPER}({prepare_parameters}):",
                    "    call = (mask, dtype, locals().get('tgt_len'))",
                    "    original_calls.append(call)",
                    "    return ('upstream', call)",
                    f"def {bootstrap._BERT_EXPAND_MASK_HELPER}({expand_parameters}):",
                    "    return ('expanded', mask)",
                )
            ),
            namespace,
        )
        setattr(
            utils,
            bootstrap._BERT_SDPA_MASK_HELPER,
            namespace[bootstrap._BERT_SDPA_MASK_HELPER],
        )
        setattr(
            utils,
            bootstrap._BERT_EXPAND_MASK_HELPER,
            namespace[bootstrap._BERT_EXPAND_MASK_HELPER],
        )
        bert = ModuleType(bootstrap._BERT_SDPA_MASK_MODULE)
        if not omit_prepare:
            setattr(
                bert,
                bootstrap._BERT_SDPA_MASK_HELPER,
                getattr(utils, bootstrap._BERT_SDPA_MASK_HELPER),
            )
        return {
            "torch": torch,
            "transformers": transformers,
            bootstrap._BERT_SDPA_MASK_UTILS_MODULE: utils,
            bootstrap._BERT_SDPA_MASK_MODULE: bert,
        }, original_calls

    def test_cpu_uses_upstream_and_enable_is_idempotent(self) -> None:
        modules, original_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            first = bootstrap.enable_webgpu_bert_sdpa_mask_compatibility()
            prepare = getattr(
                modules[bootstrap._BERT_SDPA_MASK_MODULE],
                bootstrap._BERT_SDPA_MASK_HELPER,
            )
            cpu = FakeTensor("cpu", object(), (2, 4))
            self.assertEqual(
                prepare(cpu, "float32", 3),
                ("upstream", (cpu, "float32", 3)),
            )
            second = bootstrap.enable_webgpu_bert_sdpa_mask_compatibility()
        self.assertEqual(original_calls, [(cpu, "float32", 3)])
        self.assertEqual(first["newly_patched"], 1)
        self.assertEqual(second["already_patched"], 1)
        self.assertIs(
            getattr(
                modules[bootstrap._BERT_SDPA_MASK_MODULE],
                bootstrap._BERT_SDPA_MASK_HELPER,
            ),
            prepare,
        )

    def test_mismatches_fail_closed(self) -> None:
        for modules in (
            self.make_modules(transformers_version="4.47.0")[0],
            self.make_modules(omit_prepare=True)[0],
            self.make_modules(bad_prepare_signature=True)[0],
            self.make_modules(bad_expand_signature=True)[0],
            self.make_modules(include_operators=False)[0],
        ):
            bert = modules[bootstrap._BERT_SDPA_MASK_MODULE]
            original = getattr(bert, bootstrap._BERT_SDPA_MASK_HELPER, None)
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaises(bootstrap.TransformersBrowserProfileError):
                    bootstrap.enable_webgpu_bert_sdpa_mask_compatibility()
            self.assertIs(
                getattr(bert, bootstrap._BERT_SDPA_MASK_HELPER, None),
                original,
            )


class OptSdpaMaskCompatibilityTests(unittest.TestCase):
    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        omit_class: bool = False,
        bad_signature: bool = False,
        include_ones: bool = True,
    ) -> tuple[dict[str, ModuleType], ModuleType, list[tuple[object, ...]]]:
        ones_calls: list[tuple[object, ...]] = []

        def ones(*shape: object, device: object) -> tuple[object, ...]:
            call = (*shape, device)
            ones_calls.append(call)
            return ("ones", *call)

        torch = ModuleType("torch")
        if include_ones:
            torch.ones = ones  # type: ignore[attr-defined]
        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        modules = {"torch": torch, "transformers": transformers}

        module_name, class_name = bootstrap._OPT_SDPA_MASK_TARGET
        module = ModuleType(module_name)
        if not omit_class:
            namespace: dict[str, object] = {"__name__": module_name}
            if bad_signature:
                source = "\n".join(
                    (
                        f"class {class_name}:",
                        "    def _update_causal_mask(self, inputs_embeds):",
                        "        return ('bad-upstream', inputs_embeds)",
                    )
                )
            else:
                source = "\n".join(
                    (
                        f"class {class_name}:",
                        "    def _update_causal_mask(",
                        "        self, inputs_embeds, input_shape,",
                        "        past_key_values_length, attention_mask=None,",
                        "        head_mask=None, output_attentions=None,",
                        "    ):",
                        "        call = (inputs_embeds, input_shape, past_key_values_length, attention_mask, head_mask, output_attentions)",
                        "        self.original_calls.append(call)",
                        "        return ('upstream', call)",
                    )
                )
            exec(source, namespace)
            setattr(module, class_name, namespace[class_name])
        modules[module_name] = module
        return modules, torch, ones_calls

    def make_instance(
        self,
        modules: dict[str, ModuleType],
        *,
        use_sdpa: bool = True,
        use_flash_attention_2: bool = False,
    ) -> object:
        module_name, class_name = bootstrap._OPT_SDPA_MASK_TARGET
        instance = getattr(modules[module_name], class_name)()
        instance._use_sdpa = use_sdpa
        instance._use_flash_attention_2 = use_flash_attention_2
        instance.original_calls = []
        return instance

    def test_exact_known_all_one_webgpu_cases_avoid_truth_readback(self) -> None:
        modules, torch, ones_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            diagnostics = bootstrap.enable_webgpu_opt_sdpa_mask_compatibility()

            self.assertEqual(diagnostics["newly_patched"], 1)
            self.assertEqual(diagnostics["already_patched"], 0)
            self.assertEqual(
                diagnostics["profile"],
                "transformers-4.46.3-webgpu-opt-sdpa-mask",
            )
            self.assertEqual(
                diagnostics[
                    "host_mask_truth_readbacks_avoided_per_eligible_call"
                ],
                1,
            )

            instance = self.make_instance(modules)
            webgpu = FakeTensor("webgpu", object(), (2, 4, 16))
            full = instance._update_causal_mask(webgpu, (2, 4), 0)
            self.assertIsNone(full[0])
            self.assertEqual(full[1], ("ones", 2, 4, webgpu.device))

            decode = instance._update_causal_mask(webgpu, (2, 1), 5)
            self.assertIsNone(decode[0])
            self.assertEqual(decode[1], ("ones", 2, 6, webgpu.device))
            self.assertEqual(instance.original_calls, [])

        self.assertEqual(
            ones_calls,
            [(2, 4, webgpu.device), (2, 6, webgpu.device)],
        )
        self.assertTrue(callable(torch.ones))

    def test_cpu_and_non_equivalent_webgpu_cases_use_upstream(self) -> None:
        modules, _, ones_calls = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            bootstrap.enable_webgpu_opt_sdpa_mask_compatibility()

            cpu = FakeTensor("cpu", object(), (2, 4, 16))
            webgpu = FakeTensor("webgpu", object(), (2, 4, 16))
            user_mask = object()
            head_mask = object()
            cases = (
                (self.make_instance(modules), cpu, (2, 4), 0, None, None, None),
                (
                    self.make_instance(modules),
                    webgpu,
                    (2, 4),
                    0,
                    user_mask,
                    None,
                    None,
                ),
                (self.make_instance(modules), webgpu, (2, 2), 3, None, None, None),
                (
                    self.make_instance(modules, use_sdpa=False),
                    webgpu,
                    (2, 4),
                    0,
                    None,
                    None,
                    None,
                ),
                (self.make_instance(modules), webgpu, (2, 4), 0, None, None, True),
                (
                    self.make_instance(modules),
                    webgpu,
                    (2, 4),
                    0,
                    None,
                    head_mask,
                    None,
                ),
                (
                    self.make_instance(modules, use_flash_attention_2=True),
                    webgpu,
                    (2, 4),
                    0,
                    None,
                    None,
                    None,
                ),
            )
            for instance, tensor, shape, past, mask, heads, outputs in cases:
                expected_call = (tensor, shape, past, mask, heads, outputs)
                self.assertEqual(
                    instance._update_causal_mask(
                        tensor,
                        shape,
                        past,
                        mask,
                        heads,
                        outputs,
                    ),
                    ("upstream", expected_call),
                )
                self.assertEqual(instance.original_calls, [expected_call])

        self.assertEqual(ones_calls, [])

    def test_enable_is_idempotent(self) -> None:
        modules, _, _ = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            first = bootstrap.enable_webgpu_opt_sdpa_mask_compatibility()
            module_name, class_name = bootstrap._OPT_SDPA_MASK_TARGET
            update = getattr(modules[module_name], class_name)._update_causal_mask
            second = bootstrap.enable_webgpu_opt_sdpa_mask_compatibility()

            self.assertEqual(first["newly_patched"], 1)
            self.assertEqual(second["newly_patched"], 0)
            self.assertEqual(second["already_patched"], 1)
            self.assertIs(
                getattr(modules[module_name], class_name)._update_causal_mask,
                update,
            )

    def test_version_class_signature_and_torch_ones_fail_closed(self) -> None:
        module_sets = (
            self.make_modules(transformers_version="4.47.0")[0],
            self.make_modules(omit_class=True)[0],
            self.make_modules(bad_signature=True)[0],
            self.make_modules(include_ones=False)[0],
        )
        for modules in module_sets:
            module_name, class_name = bootstrap._OPT_SDPA_MASK_TARGET
            original = (
                getattr(modules[module_name], class_name)._update_causal_mask
                if hasattr(modules[module_name], class_name)
                else None
            )
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaises(
                    bootstrap.TransformersBrowserProfileError
                ):
                    bootstrap.enable_webgpu_opt_sdpa_mask_compatibility()
            if original is not None:
                self.assertIs(
                    getattr(modules[module_name], class_name)._update_causal_mask,
                    original,
                )


class RotaryScalingCompatibilityTests(unittest.TestCase):
    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        omit_class: str | None = None,
    ) -> tuple[dict[str, ModuleType], ModuleType, list[tuple[object, float]]]:
        scalar_calls: list[tuple[object, float]] = []

        def scalar_mul(tensor: object, scalar: float) -> tuple[object, ...]:
            scalar_calls.append((tensor, scalar))
            return ("scaled", tensor, scalar)

        def no_grad():
            def decorate(function):
                @wraps(function)
                def wrapped(*args, **kwargs):
                    return function(*args, **kwargs)

                return wrapped

            return decorate

        torch = ModuleType("torch")
        torch.no_grad = no_grad  # type: ignore[attr-defined]
        torch.ops = SimpleNamespace(  # type: ignore[attr-defined]
            aten=SimpleNamespace(mul=SimpleNamespace(Scalar=scalar_mul))
        )
        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        modules = {"torch": torch, "transformers": transformers}

        for module_name, class_name in bootstrap._ROTARY_SCALING_TARGETS:
            module = ModuleType(module_name)
            if class_name != omit_class:
                namespace: dict[str, object] = {"__name__": module_name}
                exec(
                    "\n".join(
                        (
                            f"class {class_name}:",
                            "    def forward(self, x, position_ids):",
                            "        self.original_calls.append((x, position_ids))",
                            "        return ('upstream', x, position_ids)",
                        )
                    ),
                    namespace,
                )
                setattr(module, class_name, namespace[class_name])
            modules[module_name] = module
        return modules, torch, scalar_calls

    def test_exact_pinned_classes_adapt_only_webgpu_calls(self) -> None:
        modules, _, _ = self.make_modules()
        adapted = object()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                bootstrap,
                "_webgpu_rotary_embedding_forward",
                return_value=adapted,
            ) as webgpu_forward,
        ):
            diagnostics = (
                bootstrap.enable_webgpu_rotary_scaling_compatibility()
            )

            self.assertEqual(diagnostics["newly_patched"], 2)
            self.assertEqual(diagnostics["already_patched"], 0)
            self.assertEqual(diagnostics["identity_dispatches_saved_per_call"], 2)
            for module_name, class_name in bootstrap._ROTARY_SCALING_TARGETS:
                target_class = getattr(modules[module_name], class_name)
                instance = target_class()
                instance.original_calls = []
                positions = object()
                cpu_input = FakeTensor("cpu", object(), (1, 4, 16))
                self.assertEqual(
                    instance.forward(cpu_input, positions),
                    ("upstream", cpu_input, positions),
                )
                self.assertEqual(instance.original_calls, [(cpu_input, positions)])

                gpu_input = FakeTensor("webgpu", object(), (1, 4, 16))
                self.assertIs(instance.forward(gpu_input, positions), adapted)
                target_name = f"{module_name}.{class_name}"
                self.assertEqual(webgpu_forward.call_args.args[3], target_name)
                self.assertEqual(instance.original_calls, [(cpu_input, positions)])

    def test_identity_elides_both_dispatches_and_nonidentity_is_scalar(self) -> None:
        _, torch, scalar_calls = self.make_modules()
        cos = object()
        sin = object()

        self.assertEqual(
            bootstrap._apply_webgpu_rotary_scaling(cos, sin, 1.0, torch),
            (cos, sin),
        )
        self.assertEqual(scalar_calls, [])

        scaled = bootstrap._apply_webgpu_rotary_scaling(
            cos,
            sin,
            1.25,
            torch,
        )
        self.assertEqual(
            scaled,
            (("scaled", cos, 1.25), ("scaled", sin, 1.25)),
        )
        self.assertEqual(scalar_calls, [(cos, 1.25), (sin, 1.25)])

    def test_enable_is_idempotent(self) -> None:
        modules, _, _ = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            first = bootstrap.enable_webgpu_rotary_scaling_compatibility()
            forward_methods = [
                getattr(modules[module_name], class_name).forward
                for module_name, class_name in bootstrap._ROTARY_SCALING_TARGETS
            ]
            second = bootstrap.enable_webgpu_rotary_scaling_compatibility()

            self.assertEqual(first["newly_patched"], 2)
            self.assertEqual(second["newly_patched"], 0)
            self.assertEqual(second["already_patched"], 2)
            self.assertEqual(
                forward_methods,
                [
                    getattr(modules[module_name], class_name).forward
                    for module_name, class_name in bootstrap._ROTARY_SCALING_TARGETS
                ],
            )

    def test_version_and_missing_class_fail_without_partial_patch(self) -> None:
        for modules in (
            self.make_modules(transformers_version="4.47.0")[0],
            self.make_modules(omit_class="LlamaRotaryEmbedding")[0],
        ):
            originals = {
                class_name: getattr(modules[module_name], class_name).forward
                for module_name, class_name in bootstrap._ROTARY_SCALING_TARGETS
                if hasattr(modules[module_name], class_name)
            }
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaises(
                    bootstrap.TransformersBrowserProfileError
                ):
                    bootstrap.enable_webgpu_rotary_scaling_compatibility()
            for module_name, class_name in bootstrap._ROTARY_SCALING_TARGETS:
                if class_name in originals:
                    self.assertIs(
                        getattr(modules[module_name], class_name).forward,
                        originals[class_name],
                    )

    def test_invalid_attention_scaling_fails_closed(self) -> None:
        target = "transformers.models.llama.modeling_llama.LlamaRotaryEmbedding"
        for value in (True, float("inf"), object()):
            with self.assertRaisesRegex(
                bootstrap.TransformersBrowserProfileError,
                r"attention_scaling",
            ):
                bootstrap._checked_attention_scaling(value, target)


class PreallocatedKvCacheCompatibilityTests(unittest.TestCase):
    class Tensor:
        def __init__(
            self,
            shape: tuple[int, ...],
            dtype: object,
            *,
            device: str = "webgpu",
            values: list[float | int] | None = None,
            storage: list[float | int] | None = None,
            storage_shape: tuple[int, ...] | None = None,
            contiguous: bool = True,
        ) -> None:
            self.shape = shape
            self.dtype = dtype
            self.device = SimpleNamespace(type=device)
            self._contiguous = contiguous
            self.storage_shape = storage_shape or shape
            elements = 1
            for size in self.storage_shape:
                elements *= size
            self.storage = storage if storage is not None else [0.0] * elements
            if values is not None:
                self.storage[: len(values)] = values

        def is_contiguous(self) -> bool:
            return self._contiguous

        def narrow(self, dim: int, start: int, length: int):
            if dim != 2 or start != 0 or len(self.shape) != 4:
                raise AssertionError("test tensor supports prefix narrow only")
            shape = (*self.shape[:2], length, self.shape[3])
            return PreallocatedKvCacheCompatibilityTests.Tensor(
                shape,
                self.dtype,
                device=self.device.type,
                storage=self.storage,
                storage_shape=self.storage_shape,
            )

        def flat_values(self) -> list[float | int]:
            if len(self.shape) == 1:
                return list(self.storage[: self.shape[0]])
            batch, heads, sequence, features = self.shape
            capacity = self.storage_shape[2]
            result: list[float | int] = []
            for batch_idx in range(batch):
                for head in range(heads):
                    for token in range(sequence):
                        base = (
                            ((batch_idx * heads + head) * capacity + token)
                            * features
                        )
                        result.extend(self.storage[base : base + features])
            return result

    def make_modules(
        self,
        *,
        transformers_version: str = "4.46.3",
        include_op: bool = True,
        bad_signature: bool = False,
        implementation_collision: bool = False,
    ) -> tuple[
        dict[str, ModuleType],
        ModuleType,
        list[tuple[object, ...]],
        list[tuple[object, ...]],
    ]:
        float32 = object()
        int32 = object()
        int64 = object()
        allocations: list[tuple[object, ...]] = []
        updates: list[tuple[object, ...]] = []

        def empty(shape, *, dtype, device):
            allocations.append((tuple(shape), dtype, device))
            return self.Tensor(tuple(shape), dtype, device=device)

        def update_kv_cache_(
            key_cache,
            value_cache,
            key_states,
            value_states,
            positions,
        ):
            updates.append(
                (
                    key_cache,
                    value_cache,
                    key_states,
                    value_states,
                    positions,
                )
            )
            batch, heads, tokens, features = key_states.shape
            capacity = key_cache.shape[2]
            position_values = [int(value) for value in positions.flat_values()]
            key_values = key_states.flat_values()
            value_values = value_states.flat_values()
            for batch_idx in range(batch):
                for head in range(heads):
                    for token in range(tokens):
                        position = position_values[token]
                        for feature in range(features):
                            source = (
                                ((batch_idx * heads + head) * tokens + token)
                                * features
                                + feature
                            )
                            destination = (
                                (
                                    (batch_idx * heads + head) * capacity
                                    + position
                                )
                                * features
                                + feature
                            )
                            key_cache.storage[destination] = key_values[source]
                            value_cache.storage[destination] = value_values[source]

        torch = ModuleType("torch")
        torch.float32 = float32  # type: ignore[attr-defined]
        torch.int32 = int32  # type: ignore[attr-defined]
        torch.int64 = int64  # type: ignore[attr-defined]
        torch.empty = empty  # type: ignore[attr-defined]
        torch.ops = SimpleNamespace(  # type: ignore[attr-defined]
            webgpu=SimpleNamespace(
                **({"update_kv_cache_": update_kv_cache_} if include_op else {})
            )
        )

        transformers = ModuleType("transformers")
        transformers.__version__ = transformers_version  # type: ignore[attr-defined]
        cache_module_name = "transformers.cache_utils"
        cache_module = ModuleType(cache_module_name)
        namespace: dict[str, object] = {"__name__": cache_module_name}
        if bad_signature:
            source = """
class DynamicCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []
    def update(self, key_states):
        return key_states
    def get_seq_length(self, layer_idx=0):
        return 0
"""
        else:
            source = """
class DynamicCache:
    def __init__(self, num_hidden_layers=None):
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        raise AssertionError('upstream DynamicCache.update must remain untouched')
    def get_seq_length(self, layer_idx=0):
        return 0
"""
        exec(source, namespace)
        cache_module.DynamicCache = namespace["DynamicCache"]  # type: ignore[attr-defined]

        generation_config_name = (
            "transformers.generation.configuration_utils"
        )
        generation_config = ModuleType(generation_config_name)
        mapping = (
            {bootstrap._PREALLOCATED_KV_CACHE_IMPLEMENTATION: object()}
            if implementation_collision
            else {"static": object()}
        )
        generation_config.NEED_SETUP_CACHE_CLASSES_MAPPING = mapping  # type: ignore[attr-defined]
        generation_config.ALL_CACHE_IMPLEMENTATIONS = list(mapping)  # type: ignore[attr-defined]
        modules = {
            "torch": torch,
            "transformers": transformers,
            cache_module_name: cache_module,
            generation_config_name: generation_config,
        }
        return modules, torch, allocations, updates

    @staticmethod
    def config(model_type: str = "qwen2") -> SimpleNamespace:
        return SimpleNamespace(
            model_type=model_type,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
        )

    def tensor(
        self,
        torch: ModuleType,
        shape: tuple[int, ...],
        values: list[float | int],
        *,
        dtype: object | None = None,
        device: str = "webgpu",
        contiguous: bool = True,
    ) -> Tensor:
        return self.Tensor(
            shape,
            torch.float32 if dtype is None else dtype,
            device=device,
            values=values,
            contiguous=contiguous,
        )

    def test_multistep_decode_reuses_storage_and_publishes_exact_prefix(self) -> None:
        modules, torch, allocations, updates = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            diagnostics = bootstrap.enable_webgpu_preallocated_kv_cache()
            generation = modules[
                "transformers.generation.configuration_utils"
            ]
            cache_class = generation.NEED_SETUP_CACHE_CLASSES_MAPPING[  # type: ignore[attr-defined]
                bootstrap._PREALLOCATED_KV_CACHE_IMPLEMENTATION
            ]
            cache = cache_class(
                config=self.config(),
                batch_size=1,
                max_cache_len=4,
                device="webgpu",
                dtype=torch.float32,
            )
            self.assertEqual(len(allocations), 4)
            key_storage = cache._pyodide_pytorch_key_storage[0]
            value_storage = cache._pyodide_pytorch_value_storage[0]

            prefill_keys = self.tensor(
                torch,
                (1, 2, 2, 2),
                [1, 2, 3, 4, 11, 12, 13, 14],
            )
            prefill_values = self.tensor(
                torch,
                (1, 2, 2, 2),
                [21, 22, 23, 24, 31, 32, 33, 34],
            )
            prefill_positions = self.tensor(
                torch,
                (2,),
                [0, 1],
                dtype=torch.int64,
            )
            prefix_keys, prefix_values = cache.update(
                prefill_keys,
                prefill_values,
                0,
                {"cache_position": prefill_positions},
            )
            self.assertEqual(prefix_keys.shape, (1, 2, 2, 2))
            self.assertEqual(prefix_values.shape, (1, 2, 2, 2))
            self.assertEqual(prefix_keys.flat_values(), prefill_keys.flat_values())
            self.assertEqual(
                prefix_values.flat_values(), prefill_values.flat_values()
            )

            decode_keys = self.tensor(
                torch, (1, 2, 1, 2), [5, 6, 15, 16]
            )
            decode_values = self.tensor(
                torch, (1, 2, 1, 2), [25, 26, 35, 36]
            )
            decode_position = self.tensor(
                torch, (1,), [2], dtype=torch.int32
            )
            all_keys, all_values = cache.update(
                decode_keys,
                decode_values,
                0,
                {"cache_position": decode_position},
            )

            self.assertEqual(all_keys.shape, (1, 2, 3, 2))
            self.assertEqual(all_values.shape, (1, 2, 3, 2))
            self.assertEqual(
                all_keys.flat_values(),
                [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16],
            )
            self.assertEqual(
                all_values.flat_values(),
                [21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36],
            )
            self.assertIs(all_keys.storage, key_storage.storage)
            self.assertIs(all_values.storage, value_storage.storage)
            self.assertEqual(cache.get_seq_length(), 3)
            self.assertEqual(len(updates), 2)
            self.assertEqual(len(allocations), 4)
            self.assertIs(updates[1][-1], decode_position)

            self.assertEqual(
                diagnostics["dynamic_decode_dispatches_per_layer"], 4
            )
            self.assertEqual(
                diagnostics["preallocated_decode_dispatches_per_layer"], 1
            )
            self.assertEqual(
                diagnostics["dispatches_saved_per_layer_per_token"], 3
            )

    def test_update_checks_positions_shapes_capacity_and_keeps_state_on_error(self) -> None:
        modules, torch, allocations, updates = self.make_modules()
        with mock.patch.dict(sys.modules, modules):
            bootstrap.enable_webgpu_preallocated_kv_cache()
            cache_class = modules[
                "transformers.cache_utils"
            ].WebGPUPreallocatedCache  # type: ignore[attr-defined]
            cache = cache_class(
                config=self.config("llama"),
                batch_size=1,
                max_cache_len=2,
                device="webgpu",
                dtype=torch.float32,
            )
            states = self.tensor(torch, (1, 2, 1, 2), [1, 2, 3, 4])
            positions = self.tensor(
                torch, (1,), [0], dtype=torch.int64
            )

            for kwargs in (
                None,
                {"cache_position": self.tensor(
                    torch, (2,), [0, 1], dtype=torch.int64
                )},
                {"cache_position": self.tensor(
                    torch,
                    (1,),
                    [0],
                    dtype=torch.int64,
                    contiguous=False,
                )},
            ):
                with self.assertRaises(
                    bootstrap.TransformersBrowserProfileError
                ):
                    cache.update(states, states, 0, kwargs)
            self.assertEqual(cache.get_seq_length(), 0)
            self.assertEqual(updates, [])

            cache.update(
                states, states, 0, {"cache_position": positions}
            )
            cache.update(
                states,
                states,
                0,
                {"cache_position": self.tensor(
                    torch, (1,), [1], dtype=torch.int64
                )},
            )
            with self.assertRaisesRegex(
                bootstrap.TransformersBrowserProfileError, "capacity"
            ):
                cache.update(
                    states,
                    states,
                    0,
                    {"cache_position": self.tensor(
                        torch, (1,), [2], dtype=torch.int64
                    )},
                )
            self.assertEqual(cache.get_seq_length(), 2)
            self.assertEqual(len(updates), 2)
            self.assertEqual(len(allocations), 4)

            cache.reset()
            self.assertEqual(cache.get_seq_length(), 0)
            self.assertEqual(len(allocations), 4)
            self.assertEqual(cache.key_cache, [])

    def test_enable_is_idempotent_opt_in_and_cpu_dynamic_cache_is_unchanged(self) -> None:
        modules, torch, _, _ = self.make_modules()
        cache_module = modules["transformers.cache_utils"]
        dynamic_cache = cache_module.DynamicCache  # type: ignore[attr-defined]
        original_update = dynamic_cache.update
        with mock.patch.dict(sys.modules, modules):
            first = bootstrap.enable_webgpu_preallocated_kv_cache()
            second = bootstrap.enable_webgpu_preallocated_kv_cache()
            generation = modules[
                "transformers.generation.configuration_utils"
            ]

            self.assertEqual(first["newly_registered"], 1)
            self.assertEqual(second["newly_registered"], 0)
            self.assertEqual(second["already_registered"], 1)
            self.assertIs(dynamic_cache.update, original_update)
            self.assertIs(
                generation.NEED_SETUP_CACHE_CLASSES_MAPPING[  # type: ignore[attr-defined]
                    bootstrap._PREALLOCATED_KV_CACHE_IMPLEMENTATION
                ],
                cache_module.WebGPUPreallocatedCache,  # type: ignore[attr-defined]
            )
            self.assertNotIn(
                "static",
                cache_module.WebGPUPreallocatedCache.__name__.lower(),  # type: ignore[attr-defined]
            )
            with self.assertRaisesRegex(
                bootstrap.TransformersBrowserProfileError,
                "every layer on webgpu",
            ):
                cache_module.WebGPUPreallocatedCache(  # type: ignore[attr-defined]
                    config=self.config("mistral"),
                    batch_size=1,
                    max_cache_len=4,
                    device="cpu",
                    dtype=torch.float32,
                )

    def test_version_operator_signature_and_collision_fail_without_registration(self) -> None:
        module_sets = (
            self.make_modules(transformers_version="4.47.0")[0],
            self.make_modules(include_op=False)[0],
            self.make_modules(bad_signature=True)[0],
            self.make_modules(implementation_collision=True)[0],
        )
        for modules in module_sets:
            cache_module = modules["transformers.cache_utils"]
            generation = modules[
                "transformers.generation.configuration_utils"
            ]
            before_mapping = dict(  # type: ignore[attr-defined]
                generation.NEED_SETUP_CACHE_CLASSES_MAPPING
            )
            before_implementations = list(  # type: ignore[attr-defined]
                generation.ALL_CACHE_IMPLEMENTATIONS
            )
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaises(
                    bootstrap.TransformersBrowserProfileError
                ):
                    bootstrap.enable_webgpu_preallocated_kv_cache()
            self.assertFalse(
                hasattr(cache_module, bootstrap._PREALLOCATED_KV_CACHE_CLASS)
            )
            self.assertEqual(  # type: ignore[attr-defined]
                generation.NEED_SETUP_CACHE_CLASSES_MAPPING,
                before_mapping,
            )
            self.assertEqual(  # type: ignore[attr-defined]
                generation.ALL_CACHE_IMPLEMENTATIONS,
                before_implementations,
            )


if __name__ == "__main__":
    unittest.main()
