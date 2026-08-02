"""Exact-version Gemma2 compatibility for the browser WebGPU backend.

Transformers 4.46.3 stores ``Gemma2RMSNorm.weight`` as an offset and computes
``rms(x) * (1 + weight)``. Ordinary ``torch.nn.functional.rms_norm`` instead
treats its weight as the complete multiplier, so the general decoder adapter
must not patch this class. This module installs a narrow, version-checked
adapter backed by ``webgpu::gemma_rms_norm`` and leaves every unsupported call
on the captured upstream method.

Mask construction is deliberately not rewritten here. The corresponding ATen
operators are reusable backend primitives and preserve upstream Gemma2 code.
"""

from __future__ import annotations

import importlib
import inspect
import math
from functools import wraps
from typing import Any, Callable


_SUPPORTED_TRANSFORMERS_VERSION = "4.46.3"
_GEMMA2_MODULE = "transformers.models.gemma2.modeling_gemma2"
_GEMMA2_RMS_NORM_CLASS = "Gemma2RMSNorm"
_GEMMA2_RMS_NORM_MARKER = "_pyodide_pytorch_webgpu_gemma2_rms_norm_target"
_GEMMA2_MODEL_CLASS = "Gemma2Model"
_GEMMA2_NORMALIZER_MARKER = (
    "_pyodide_pytorch_webgpu_gemma2_scalar_normalizer_target"
)
_GEMMA2_NORMALIZER_CACHE = "_pyodide_pytorch_webgpu_gemma2_normalizer"


class Gemma2WebGPUProfileError(RuntimeError):
    """The pinned Gemma2 WebGPU adapter cannot be enabled safely."""


class _DeviceLocalTensorProxy:
    """Delegate torch except for Gemma2's one verified scalar construction."""

    def __init__(
        self,
        torch: Any,
        expected_value: float,
        expected_dtype: Any,
        replacement: Any,
    ) -> None:
        self._torch = torch
        self._expected_value = expected_value
        self._expected_dtype = expected_dtype
        self._replacement = replacement
        self.replacements = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def tensor(self, *args: Any, **kwargs: Any) -> Any:
        if (
            len(args) == 1
            and args[0] == self._expected_value
            and kwargs == {"dtype": self._expected_dtype}
        ):
            self.replacements += 1
            if self.replacements != 1:
                raise Gemma2WebGPUProfileError(
                    "pinned Gemma2 constructed its embedding normalizer more "
                    "than once; review the upstream forward before continuing"
                )
            return self._replacement
        return self._torch.tensor(*args, **kwargs)


def _eligible(module: Any, hidden_states: Any, torch: Any) -> bool:
    try:
        weight = module.weight
        if not (
            hidden_states.device.type == "webgpu"
            and weight.device == hidden_states.device
            and hidden_states.dtype == torch.float32
            and weight.dtype == torch.float32
            and hidden_states.ndim >= 1
            and weight.ndim == 1
            and weight.shape[0] == hidden_states.shape[-1]
            and weight.is_contiguous()
        ):
            return False
        is_grad_enabled = getattr(torch, "is_grad_enabled", None)
        if callable(is_grad_enabled) and is_grad_enabled() and (
            bool(getattr(hidden_states, "requires_grad", False))
            or bool(getattr(weight, "requires_grad", False))
        ):
            return False
        epsilon = float(module.eps)
        return math.isfinite(epsilon) and epsilon >= 0.0
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        return False


def _make_forward(
    original_forward: Callable[[Any, Any], Any],
    target_name: str,
    torch: Any,
) -> Callable[[Any, Any], Any]:
    @wraps(original_forward)
    def webgpu_gemma2_rms_norm_forward(self: Any, x: Any) -> Any:
        if not _eligible(self, x, torch):
            return original_forward(self, x)
        return torch.ops.webgpu.gemma_rms_norm(
            x,
            self.weight,
            float(self.eps),
        )

    setattr(
        webgpu_gemma2_rms_norm_forward,
        _GEMMA2_RMS_NORM_MARKER,
        target_name,
    )
    return webgpu_gemma2_rms_norm_forward


def enable_webgpu_gemma2_rms_norm() -> dict[str, object]:
    """Fuse pinned Gemma2's offset-weight RMSNorm on float32 WebGPU.

    The exact Transformers version, canonical class identity, Python method
    signature, and custom operator are checked before mutation. Repeated calls
    are idempotent. CPU, non-float32, malformed, and autograd calls execute the
    captured Transformers method unchanged.
    """

    try:
        import torch
        import transformers
    except ImportError as error:
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU RMSNorm requires torch and Transformers; no "
            "classes were changed"
        ) from error

    installed_version = getattr(transformers, "__version__", "<unknown>")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU RMSNorm supports exactly Transformers "
            f"{_SUPPORTED_TRANSFORMERS_VERSION}; found {installed_version}. "
            "No classes were changed. Review Gemma2's normalization contract "
            "before extending the adapter."
        )
    custom_op = getattr(
        getattr(getattr(torch, "ops", None), "webgpu", None),
        "gemma_rms_norm",
        None,
    )
    if not callable(custom_op):
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU RMSNorm requires "
            "torch.ops.webgpu.gemma_rms_norm; no classes were changed"
        )

    try:
        module = importlib.import_module(_GEMMA2_MODULE)
    except Exception as error:
        raise Gemma2WebGPUProfileError(
            f"Gemma2 WebGPU RMSNorm could not import {_GEMMA2_MODULE}: "
            f"{error}. No classes were changed."
        ) from error
    target_class = getattr(module, _GEMMA2_RMS_NORM_CLASS, None)
    target_name = f"{_GEMMA2_MODULE}.{_GEMMA2_RMS_NORM_CLASS}"
    if not isinstance(target_class, type):
        raise Gemma2WebGPUProfileError(
            f"Gemma2 WebGPU RMSNorm expected class {target_name}; no classes "
            "were changed"
        )
    if (
        target_class.__module__ != _GEMMA2_MODULE
        or target_class.__name__ != _GEMMA2_RMS_NORM_CLASS
    ):
        raise Gemma2WebGPUProfileError(
            f"Gemma2 WebGPU RMSNorm found an unexpected class exported as "
            f"{target_name}; no classes were changed"
        )

    forward = getattr(target_class, "forward", None)
    marker = getattr(forward, _GEMMA2_RMS_NORM_MARKER, None)
    if marker is not None and marker != target_name:
        raise Gemma2WebGPUProfileError(
            f"Gemma2 WebGPU RMSNorm found an incompatible adapter on "
            f"{target_name}; no classes were changed"
        )
    already_enabled = marker == target_name
    if not already_enabled:
        if not inspect.isfunction(forward):
            raise Gemma2WebGPUProfileError(
                f"Gemma2 WebGPU RMSNorm expected a Python forward method on "
                f"{target_name}; no classes were changed"
            )
        parameter_names = tuple(inspect.signature(forward).parameters)
        if (
            forward.__module__ != _GEMMA2_MODULE
            or forward.__qualname__
            != f"{_GEMMA2_RMS_NORM_CLASS}.forward"
            or parameter_names != ("self", "x")
        ):
            raise Gemma2WebGPUProfileError(
                "Gemma2 WebGPU RMSNorm found an unexpected forward "
                f"implementation on {target_name}; no classes were changed"
            )
        target_class.forward = _make_forward(  # type: ignore[method-assign]
            forward,
            target_name,
            torch,
        )

    return {
        "enabled": True,
        "profile": "transformers-4.46.3-webgpu-gemma2-rms-norm",
        "transformers_version": installed_version,
        "supported_transformers_version": _SUPPORTED_TRANSFORMERS_VERSION,
        "newly_patched": 0 if already_enabled else 1,
        "already_patched": 1 if already_enabled else 0,
        "upstream_dispatches_per_call": 6,
        "fused_dispatches_per_call": 1,
        "dispatches_saved_per_call": 5,
        "two_layer_fixture_calls": 9,
        "two_layer_fixture_dispatches_saved": 45,
        "targets": [
            {
                "target": target_name,
                "status": "already_enabled" if already_enabled else "patched",
            }
        ],
    }


def _make_model_forward(
    original_forward: Callable[..., Any],
    target_name: str,
    modeling_module: Any,
    torch: Any,
) -> Callable[..., Any]:
    signature = inspect.signature(original_forward)

    @wraps(original_forward)
    def webgpu_gemma2_model_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            arguments = signature.bind(self, *args, **kwargs)
            arguments.apply_defaults()
            inputs_embeds = arguments.arguments["inputs_embeds"]
            input_ids = arguments.arguments["input_ids"]
            input_tensor = (
                inputs_embeds if inputs_embeds is not None else input_ids
            )
            embedding_weight = self.embed_tokens.weight
            eligible = (
                input_tensor is not None
                and input_tensor.device.type == "webgpu"
                and embedding_weight.device.type == "webgpu"
                and embedding_weight.dtype == torch.float32
                and (
                    inputs_embeds is None
                    or inputs_embeds.dtype == torch.float32
                )
            )
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            eligible = False
        if not eligible:
            return original_forward(self, *args, **kwargs)

        expected_value = self.config.hidden_size**0.5
        cache = getattr(self, _GEMMA2_NORMALIZER_CACHE, None)
        if not (
            isinstance(cache, tuple)
            and len(cache) == 2
            and cache[0] == expected_value
            and cache[1].device == embedding_weight.device
            and cache[1].dtype == embedding_weight.dtype
            and tuple(cache[1].shape) == ()
        ):
            cached_tensor = torch.full(
                (),
                expected_value,
                dtype=embedding_weight.dtype,
                device=embedding_weight.device,
            )
            # A plain private tuple intentionally stays outside state_dict.
            # Device or dtype changes invalidate it on the next call.
            object.__setattr__(
                self,
                _GEMMA2_NORMALIZER_CACHE,
                (expected_value, cached_tensor),
            )
        else:
            cached_tensor = cache[1]

        original_torch = modeling_module.torch
        if original_torch is not torch:
            raise Gemma2WebGPUProfileError(
                "pinned Gemma2's module-global torch binding changed after "
                "the adapter was enabled"
            )
        proxy = _DeviceLocalTensorProxy(
            torch,
            expected_value,
            embedding_weight.dtype,
            cached_tensor,
        )
        modeling_module.torch = proxy
        try:
            result = original_forward(self, *args, **kwargs)
        finally:
            modeling_module.torch = original_torch
        if proxy.replacements != 1:
            raise Gemma2WebGPUProfileError(
                "pinned Gemma2 did not execute its expected embedding "
                "normalizer construction; review the upstream forward"
            )
        return result

    setattr(
        webgpu_gemma2_model_forward,
        _GEMMA2_NORMALIZER_MARKER,
        target_name,
    )
    return webgpu_gemma2_model_forward


def enable_webgpu_gemma2_scalar_normalizer() -> dict[str, object]:
    """Keep Gemma2's embedding normalizer on WebGPU.

    Pinned Transformers creates ``torch.tensor(sqrt(hidden_size))`` without a
    device inside ``Gemma2Model.forward``. With WebGPU hidden states that would
    make the subsequent ``aten::mul.Tensor`` mixed-device. The adapter caches a
    scalar on the embedding weight's WebGPU device and substitutes it for only
    that exact scalar construction while the captured synchronous forward is
    running. The module-global binding is restored in ``finally`` and the
    successful branch verifies exactly one substitution.
    """

    try:
        import torch
        import transformers
    except ImportError as error:
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU scalar normalization requires torch and "
            "Transformers; no classes were changed"
        ) from error
    installed_version = getattr(transformers, "__version__", "<unknown>")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU scalar normalization supports exactly "
            f"Transformers {_SUPPORTED_TRANSFORMERS_VERSION}; found "
            f"{installed_version}. No classes were changed."
        )
    if not callable(getattr(torch, "full", None)):
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU scalar normalization requires torch.full; no "
            "classes were changed"
        )
    try:
        modeling_module = importlib.import_module(_GEMMA2_MODULE)
    except Exception as error:
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU scalar normalization could not import "
            f"{_GEMMA2_MODULE}: {error}. No classes were changed."
        ) from error
    if getattr(modeling_module, "torch", None) is not torch:
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU scalar normalization found an unexpected "
            "module-global torch binding; no classes were changed"
        )
    target_class = getattr(modeling_module, _GEMMA2_MODEL_CLASS, None)
    target_name = f"{_GEMMA2_MODULE}.{_GEMMA2_MODEL_CLASS}"
    if not isinstance(target_class, type):
        raise Gemma2WebGPUProfileError(
            f"Gemma2 WebGPU scalar normalization expected {target_name}; "
            "no classes were changed"
        )
    if (
        target_class.__module__ != _GEMMA2_MODULE
        or target_class.__name__ != _GEMMA2_MODEL_CLASS
    ):
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU scalar normalization found an unexpected class "
            f"exported as {target_name}; no classes were changed"
        )
    forward = getattr(target_class, "forward", None)
    marker = getattr(forward, _GEMMA2_NORMALIZER_MARKER, None)
    if marker is not None and marker != target_name:
        raise Gemma2WebGPUProfileError(
            "Gemma2 WebGPU scalar normalization found an incompatible "
            f"adapter on {target_name}; no classes were changed"
        )
    already_enabled = marker == target_name
    if not already_enabled:
        expected_parameters = (
            "self",
            "input_ids",
            "attention_mask",
            "position_ids",
            "past_key_values",
            "inputs_embeds",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "cache_position",
        )
        if not inspect.isfunction(forward) or (
            forward.__module__ != _GEMMA2_MODULE
            or forward.__qualname__ != f"{_GEMMA2_MODEL_CLASS}.forward"
            or tuple(inspect.signature(forward).parameters)
            != expected_parameters
        ):
            raise Gemma2WebGPUProfileError(
                "Gemma2 WebGPU scalar normalization found an unexpected "
                f"forward implementation on {target_name}; no classes were "
                "changed"
            )
        target_class.forward = _make_model_forward(  # type: ignore[method-assign]
            forward,
            target_name,
            modeling_module,
            torch,
        )

    return {
        "enabled": True,
        "profile": "transformers-4.46.3-webgpu-gemma2-scalar-normalizer",
        "transformers_version": installed_version,
        "supported_transformers_version": _SUPPORTED_TRANSFORMERS_VERSION,
        "newly_patched": 0 if already_enabled else 1,
        "already_patched": 1 if already_enabled else 0,
        "mixed_device_mul_avoided_per_forward": 1,
        "device_scalar_allocations_per_model": "at-most-one-per-device-dtype",
        "targets": [
            {
                "target": target_name,
                "status": "already_enabled" if already_enabled else "patched",
            }
        ],
    }
