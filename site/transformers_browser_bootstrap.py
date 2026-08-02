"""Opt-in Transformers compatibility helpers for the browser runtime.

Transformers 4.46.3 imports its optional GGUF integration while importing any
configuration class. That integration imports the compiled ``tokenizers``
package unconditionally even when Transformers has already detected that the
package is unavailable. Tiny model construction does not use GGUF or a
tokenizer, so this bootstrap makes that one optional loader fail lazily.

The same pinned release implements Qwen2, Llama, Mistral, and Phi-3 RMSNorm as
six eager tensor operations. The browser WebGPU backend has an
``aten::rms_norm`` kernel, so :func:`enable_webgpu_rms_norm_fusion` installs a
version-checked adapter for those four exact upstream classes. Decode-time
Qwen2, Llama, and Mistral MLPs can similarly opt into one fused gate/up GEMV
plus SwiGLU dispatch. Their greedy decode path can opt into a fixed-capacity
DynamicCache-compatible class whose fused indexed update writes K and V
together and returns only the valid prefix to attention.
Its Qwen2 and Llama rotary classes also route numeric attention scaling through
the scalar ATen overload on WebGPU and elide the common identity scale. CPU and
unsupported calls still execute the captured upstream methods.
OPT can opt into a shape-proven SDPA-mask branch that avoids synchronously
reading an internally generated all-one WebGPU mask.

The upstream wheel and its dependency metadata remain unchanged. This module
does not create a fake ``tokenizers`` package, alter model state, or claim that
tokenization or arbitrary model implementations work.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import inspect
import math
import sys
from functools import wraps
from types import ModuleType
from typing import Any, Callable, NoReturn


_GGUF_MODULE = "transformers.modeling_gguf_pytorch_utils"
_MODEL_ONLY_MARKER = "_pyodide_pytorch_model_only"
_SUPPORTED_TRANSFORMERS_VERSION = "4.46.3"
_RMS_NORM_MARKER = "_pyodide_pytorch_webgpu_rms_norm_target"
_RMS_NORM_TARGETS = (
    (
        "transformers.models.qwen2.modeling_qwen2",
        "Qwen2RMSNorm",
    ),
    (
        "transformers.models.llama.modeling_llama",
        "LlamaRMSNorm",
    ),
    (
        "transformers.models.mistral.modeling_mistral",
        "MistralRMSNorm",
    ),
    (
        "transformers.models.phi3.modeling_phi3",
        "Phi3RMSNorm",
    ),
)
_ROTARY_SCALING_MARKER = "_pyodide_pytorch_webgpu_rotary_scaling_target"
_ROTARY_SCALING_TARGETS = (
    (
        "transformers.models.qwen2.modeling_qwen2",
        "Qwen2RotaryEmbedding",
    ),
    (
        "transformers.models.llama.modeling_llama",
        "LlamaRotaryEmbedding",
    ),
)
_SWIGLU_MARKER = "_pyodide_pytorch_webgpu_swiglu_target"
_SWIGLU_TARGETS = (
    (
        "transformers.models.qwen2.modeling_qwen2",
        "Qwen2MLP",
        "hidden_state",
    ),
    (
        "transformers.models.llama.modeling_llama",
        "LlamaMLP",
        "x",
    ),
    (
        "transformers.models.mistral.modeling_mistral",
        "MistralMLP",
        "hidden_state",
    ),
)
_PREALLOCATED_KV_CACHE_MARKER = (
    "_pyodide_pytorch_webgpu_preallocated_kv_cache"
)
_PREALLOCATED_KV_CACHE_CLASS = "WebGPUPreallocatedCache"
_PREALLOCATED_KV_CACHE_IMPLEMENTATION = "webgpu_preallocated"
_PREALLOCATED_KV_CACHE_MODEL_TYPES = frozenset(
    ("qwen2", "llama", "mistral")
)
_OPT_SDPA_MASK_MARKER = "_pyodide_pytorch_webgpu_opt_sdpa_mask_target"
_OPT_SDPA_MASK_TARGET = (
    "transformers.models.opt.modeling_opt",
    "OPTDecoder",
)


class TransformersBrowserProfileError(RuntimeError):
    """The requested browser adapter is incompatible with this installation."""


def _webgpu_device_type(device: object) -> str | None:
    if isinstance(device, str):
        return device.partition(":")[0]
    value = getattr(device, "type", None)
    return value if isinstance(value, str) else None


def _make_webgpu_preallocated_cache_class(
    dynamic_cache_class: type[Any],
    torch: ModuleType,
    module_name: str,
) -> type[Any]:
    """Build the exact-version cache class without importing Transformers here."""

    class WebGPUPreallocatedCache(dynamic_cache_class):  # type: ignore[misc, valid-type]
        """Fixed-storage, valid-prefix cache for sequential WebGPU decoding."""

        def __init__(
            self,
            config: Any,
            batch_size: int | None = None,
            max_cache_len: int | None = None,
            device: Any = None,
            dtype: Any = None,
            max_batch_size: int | None = None,
            layer_device_map: dict[int, Any] | None = None,
        ) -> None:
            super().__init__()
            resolved_batch_size = batch_size or max_batch_size
            if type(resolved_batch_size) is not int or resolved_batch_size <= 0:
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache requires a positive integer "
                    "batch_size"
                )
            if type(max_cache_len) is not int or max_cache_len <= 0:
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache requires a positive integer "
                    "max_cache_len"
                )
            model_type = getattr(config, "model_type", None)
            if model_type not in _PREALLOCATED_KV_CACHE_MODEL_TYPES:
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache supports only the pinned "
                    "Qwen2, Llama, and Mistral decoder cache contract; found "
                    f"model_type={model_type!r}"
                )

            num_layers = getattr(config, "num_hidden_layers", None)
            num_attention_heads = getattr(
                config, "num_attention_heads", None
            )
            hidden_size = getattr(config, "hidden_size", None)
            num_key_value_heads = getattr(
                config, "num_key_value_heads", None
            )
            if num_key_value_heads is None:
                num_key_value_heads = num_attention_heads
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                if (
                    type(hidden_size) is not int
                    or type(num_attention_heads) is not int
                    or num_attention_heads <= 0
                    or hidden_size <= 0
                    or hidden_size % num_attention_heads != 0
                ):
                    raise TransformersBrowserProfileError(
                        "WebGPU preallocated KV cache could not derive an "
                        "integer attention head dimension"
                    )
                head_dim = hidden_size // num_attention_heads
            integer_fields = {
                "num_hidden_layers": num_layers,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
            }
            for field, value in integer_fields.items():
                if type(value) is not int or value <= 0:
                    raise TransformersBrowserProfileError(
                        "WebGPU preallocated KV cache requires a positive "
                        f"integer config.{field}"
                    )

            resolved_dtype = torch.float32 if dtype is None else dtype
            if resolved_dtype != torch.float32:
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache currently supports only "
                    "torch.float32"
                )
            devices: list[Any] = []
            for layer_idx in range(num_layers):
                layer_device = (
                    device
                    if layer_device_map is None
                    else layer_device_map.get(layer_idx)
                )
                if _webgpu_device_type(layer_device) != "webgpu":
                    raise TransformersBrowserProfileError(
                        "WebGPU preallocated KV cache requires every layer "
                        f"on webgpu; layer {layer_idx} uses {layer_device!r}"
                    )
                devices.append(layer_device)

            self.batch_size = resolved_batch_size
            self.max_cache_len = max_cache_len
            self.head_dim = head_dim
            self.dtype = resolved_dtype
            self.num_key_value_heads = num_key_value_heads
            self._pyodide_pytorch_valid_lengths = [0] * num_layers
            cache_shape = (
                resolved_batch_size,
                num_key_value_heads,
                max_cache_len,
                head_dim,
            )
            # Empty avoids a capacity-sized fill dispatch. Invalid capacity is
            # never exposed: update publishes only a narrow valid-prefix view.
            self._pyodide_pytorch_key_storage = [
                torch.empty(cache_shape, dtype=resolved_dtype, device=layer_device)
                for layer_device in devices
            ]
            self._pyodide_pytorch_value_storage = [
                torch.empty(cache_shape, dtype=resolved_dtype, device=layer_device)
                for layer_device in devices
            ]
            self.key_cache = []
            self.value_cache = []
            self._seen_tokens = 0

        def _publish_prefix(self, layer_idx: int, length: int) -> tuple[Any, Any]:
            keys = self._pyodide_pytorch_key_storage[layer_idx].narrow(
                2, 0, length
            )
            values = self._pyodide_pytorch_value_storage[layer_idx].narrow(
                2, 0, length
            )
            while len(self.key_cache) < layer_idx:
                self.key_cache.append([])
                self.value_cache.append([])
            if len(self.key_cache) == layer_idx:
                self.key_cache.append(keys)
                self.value_cache.append(values)
            else:
                self.key_cache[layer_idx] = keys
                self.value_cache[layer_idx] = values
            return keys, values

        def update(
            self,
            key_states: Any,
            value_states: Any,
            layer_idx: int,
            cache_kwargs: dict[str, Any] | None = None,
        ) -> tuple[Any, Any]:
            if type(layer_idx) is not int or not (
                0 <= layer_idx < len(self._pyodide_pytorch_valid_lengths)
            ):
                raise TransformersBrowserProfileError(
                    f"WebGPU preallocated KV cache layer index {layer_idx!r} "
                    "is out of range"
                )
            if not isinstance(cache_kwargs, dict):
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache requires cache_kwargs with "
                    "a cache_position tensor"
                )
            cache_position = cache_kwargs.get("cache_position")
            try:
                state_shape = tuple(key_states.shape)
                value_shape = tuple(value_states.shape)
                position_shape = tuple(cache_position.shape)
                eligible = (
                    key_states.device.type == "webgpu"
                    and value_states.device.type == "webgpu"
                    and cache_position.device.type == "webgpu"
                    and key_states.dtype == torch.float32
                    and value_states.dtype == torch.float32
                    and cache_position.dtype in (torch.int32, torch.int64)
                    and len(state_shape) == 4
                    and value_shape == state_shape
                    and len(position_shape) == 1
                    and position_shape[0] == state_shape[2]
                    and cache_position.is_contiguous()
                    and state_shape[0] == self.batch_size
                    and state_shape[1] == self.num_key_value_heads
                    and state_shape[3] == self.head_dim
                )
            except (AttributeError, IndexError, TypeError, RuntimeError):
                eligible = False
            if not eligible:
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache update requires matching "
                    "float32 [batch, kv_heads, tokens, head_dim] WebGPU states "
                    "and a contiguous WebGPU int32/int64 cache_position vector"
                )

            previous_length = self._pyodide_pytorch_valid_lengths[layer_idx]
            token_count = state_shape[2]
            next_length = previous_length + token_count
            if token_count <= 0:
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache update requires at least "
                    "one token"
                )
            if next_length > self.max_cache_len:
                raise TransformersBrowserProfileError(
                    "WebGPU preallocated KV cache capacity exceeded: "
                    f"{next_length} > {self.max_cache_len}"
                )

            # The pinned DynamicCache contract is sequential append. During
            # generation cache_position is derived from get_seq_length(); the
            # GPU operator uses those positions directly without a readback.
            torch.ops.webgpu.update_kv_cache_(
                self._pyodide_pytorch_key_storage[layer_idx],
                self._pyodide_pytorch_value_storage[layer_idx],
                key_states,
                value_states,
                cache_position,
            )
            self._pyodide_pytorch_valid_lengths[layer_idx] = next_length
            if layer_idx == 0:
                self._seen_tokens = next_length
            return self._publish_prefix(layer_idx, next_length)

        def get_seq_length(self, layer_idx: int | None = 0) -> int:
            index = 0 if layer_idx is None else layer_idx
            if type(index) is not int or not (
                0 <= index < len(self._pyodide_pytorch_valid_lengths)
            ):
                return 0
            return self._pyodide_pytorch_valid_lengths[index]

        def get_max_cache_shape(self) -> int:
            return self.max_cache_len

        def reset(self) -> None:
            self._pyodide_pytorch_valid_lengths = [
                0
            ] * len(self._pyodide_pytorch_valid_lengths)
            self.key_cache = []
            self.value_cache = []
            self._seen_tokens = 0

        def crop(self, max_length: int) -> None:
            current_length = self.get_seq_length()
            requested = (
                current_length - abs(max_length)
                if max_length < 0
                else max_length
            )
            requested = max(0, min(current_length, requested))
            for layer_idx, layer_length in enumerate(
                self._pyodide_pytorch_valid_lengths
            ):
                next_length = min(layer_length, requested)
                self._pyodide_pytorch_valid_lengths[layer_idx] = next_length
                if layer_idx < len(self.key_cache):
                    self._publish_prefix(layer_idx, next_length)
            self._seen_tokens = requested

        def _unsupported_cache_transform(self, operation: str) -> NoReturn:
            raise TransformersBrowserProfileError(
                "WebGPU preallocated KV cache currently supports greedy or "
                f"sampling decode, not {operation}"
            )

        def reorder_cache(self, beam_idx: Any) -> NoReturn:
            del beam_idx
            self._unsupported_cache_transform("beam reordering")

        def batch_repeat_interleave(self, repeats: int) -> NoReturn:
            del repeats
            self._unsupported_cache_transform("batch repeat-interleave")

        def batch_select_indices(self, indices: Any) -> NoReturn:
            del indices
            self._unsupported_cache_transform("batch index selection")

        def batch_split(self, *args: Any, **kwargs: Any) -> NoReturn:
            del args, kwargs
            self._unsupported_cache_transform("batch splitting")

        @classmethod
        def from_batch_splits(cls, *args: Any, **kwargs: Any) -> NoReturn:
            del cls, args, kwargs
            raise TransformersBrowserProfileError(
                "WebGPU preallocated KV cache does not support merging batch "
                "splits"
            )

        @classmethod
        def from_legacy_cache(cls, *args: Any, **kwargs: Any) -> NoReturn:
            del cls, args, kwargs
            raise TransformersBrowserProfileError(
                "WebGPU preallocated KV cache must be created with explicit "
                "capacity; legacy conversion is unsupported"
            )

    WebGPUPreallocatedCache.__name__ = _PREALLOCATED_KV_CACHE_CLASS
    WebGPUPreallocatedCache.__qualname__ = _PREALLOCATED_KV_CACHE_CLASS
    WebGPUPreallocatedCache.__module__ = module_name
    setattr(WebGPUPreallocatedCache, _PREALLOCATED_KV_CACHE_MARKER, True)
    return WebGPUPreallocatedCache


def _preallocated_kv_diagnostics(
    installed_version: str,
    *,
    newly_registered: int,
) -> dict[str, object]:
    return {
        "enabled": True,
        "profile": "transformers-4.46.3-webgpu-preallocated-kv",
        "transformers_version": installed_version,
        "supported_transformers_version": _SUPPORTED_TRANSFORMERS_VERSION,
        "newly_registered": newly_registered,
        "already_registered": 1 - newly_registered,
        "cache_implementation": _PREALLOCATED_KV_CACHE_IMPLEMENTATION,
        "dynamic_decode_dispatches_per_layer": 4,
        "preallocated_decode_dispatches_per_layer": 1,
        "dispatches_saved_per_layer_per_token": 3,
        "dynamic_bytes_per_token_element_at_prefix_length": (
            "16 * prefix_length"
        ),
        "preallocated_bytes_per_token_element": 16,
    }


def enable_webgpu_preallocated_kv_cache() -> dict[str, object]:
    """Register the pinned one-dispatch preallocated decode cache.

    The new ``cache_implementation="webgpu_preallocated"`` option is opt-in
    and supports the exact Transformers 4.46.3 Qwen2/Llama/Mistral sequential
    cache contract. It subclasses ``DynamicCache`` deliberately: attention
    receives a narrow valid-prefix view rather than the full allocation, so
    SDPA never scores or reads unused capacity. CPU caches and the upstream
    DynamicCache/StaticCache classes are unchanged.
    """

    try:
        import torch
        import transformers
    except ImportError as error:
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache requires torch and Transformers to "
            "be installed before it is enabled; no classes were changed"
        ) from error

    installed_version = getattr(transformers, "__version__", "<unknown>")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache supports exactly Transformers "
            f"{_SUPPORTED_TRANSFORMERS_VERSION}; found {installed_version}. "
            "No classes were changed. Review cache update and generation "
            "contracts before extending the adapter."
        )
    update_op = getattr(
        getattr(getattr(torch, "ops", None), "webgpu", None),
        "update_kv_cache_",
        None,
    )
    if not callable(update_op) or not callable(getattr(torch, "empty", None)):
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache requires torch.empty and "
            "torch.ops.webgpu.update_kv_cache_; no classes were changed"
        )

    cache_module_name = "transformers.cache_utils"
    generation_config_name = "transformers.generation.configuration_utils"
    try:
        cache_module = importlib.import_module(cache_module_name)
        generation_config = importlib.import_module(generation_config_name)
    except Exception as error:
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache could not import the pinned cache "
            f"modules: {error}. No classes were changed."
        ) from error
    dynamic_cache = getattr(cache_module, "DynamicCache", None)
    if not isinstance(dynamic_cache, type) or (
        dynamic_cache.__module__ != cache_module_name
        or dynamic_cache.__name__ != "DynamicCache"
    ):
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache found an unexpected DynamicCache "
            "class; no classes were changed"
        )
    expected_signatures = {
        "__init__": ("self", "num_hidden_layers"),
        "update": (
            "self",
            "key_states",
            "value_states",
            "layer_idx",
            "cache_kwargs",
        ),
        "get_seq_length": ("self", "layer_idx"),
    }
    for method_name, expected in expected_signatures.items():
        method = getattr(dynamic_cache, method_name, None)
        if not callable(method) or tuple(
            inspect.signature(method).parameters
        ) != expected:
            raise TransformersBrowserProfileError(
                "WebGPU preallocated KV cache found an unexpected "
                f"DynamicCache.{method_name} signature; no classes were "
                "changed"
            )

    mapping = getattr(
        generation_config, "NEED_SETUP_CACHE_CLASSES_MAPPING", None
    )
    implementations = getattr(
        generation_config, "ALL_CACHE_IMPLEMENTATIONS", None
    )
    if not isinstance(mapping, dict) or not isinstance(implementations, list):
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache found unexpected generation cache "
            "registries; no classes were changed"
        )
    generation_utils = sys.modules.get("transformers.generation.utils")
    if generation_utils is not None and getattr(
        generation_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING", None
    ) is not mapping:
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache found a detached generation cache "
            "registry; no classes were changed"
        )

    existing_class = getattr(
        cache_module, _PREALLOCATED_KV_CACHE_CLASS, None
    )
    mapped_class = mapping.get(_PREALLOCATED_KV_CACHE_IMPLEMENTATION)
    if existing_class is not None or mapped_class is not None:
        if (
            existing_class is mapped_class
            and isinstance(existing_class, type)
            and getattr(
                existing_class, _PREALLOCATED_KV_CACHE_MARKER, False
            )
            and _PREALLOCATED_KV_CACHE_IMPLEMENTATION in implementations
        ):
            return _preallocated_kv_diagnostics(
                installed_version, newly_registered=0
            )
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache registration collides with an "
            "existing class or implementation; no classes were changed"
        )
    if _PREALLOCATED_KV_CACHE_IMPLEMENTATION in implementations:
        raise TransformersBrowserProfileError(
            "WebGPU preallocated KV cache implementation name is already "
            "reserved; no classes were changed"
        )

    cache_class = _make_webgpu_preallocated_cache_class(
        dynamic_cache, torch, cache_module_name
    )
    setattr(cache_module, _PREALLOCATED_KV_CACHE_CLASS, cache_class)
    mapping[_PREALLOCATED_KV_CACHE_IMPLEMENTATION] = cache_class
    implementations.append(_PREALLOCATED_KV_CACHE_IMPLEMENTATION)
    return _preallocated_kv_diagnostics(
        installed_version, newly_registered=1
    )


def disable_optional_gguf_without_tokenizers() -> bool:
    """Disable Transformers' optional GGUF loader when tokenizers is absent.

    Returns ``True`` when the model-only shim is active and ``False`` when a
    real tokenizers package is available and no shim is necessary. Repeated
    calls are safe.
    """

    if importlib.util.find_spec("tokenizers") is not None:
        return False

    existing = sys.modules.get(_GGUF_MODULE)
    if existing is not None:
        if getattr(existing, _MODEL_ONLY_MARKER, False):
            return True
        raise RuntimeError(
            "install the model-only bootstrap before importing a Transformers "
            "configuration or model class"
        )

    module = ModuleType(_GGUF_MODULE)

    def load_gguf_checkpoint(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise ImportError(
            "GGUF loading is disabled in the model-only browser stack because "
            "the compiled tokenizers package is unavailable"
        )

    module.__package__ = "transformers"
    module.__spec__ = importlib.machinery.ModuleSpec(_GGUF_MODULE, loader=None)
    module.load_gguf_checkpoint = load_gguf_checkpoint  # type: ignore[attr-defined]
    module.__all__ = ["load_gguf_checkpoint"]
    setattr(module, _MODEL_ONLY_MARKER, True)
    sys.modules[_GGUF_MODULE] = module
    return True


def _rms_norm_target_name(module_name: str, class_name: str) -> str:
    return f"{module_name}.{class_name}"


def _make_webgpu_rms_norm_forward(
    original_forward: Callable[[Any, Any], Any],
    target_name: str,
    torch: ModuleType,
) -> Callable[[Any, Any], Any]:
    @wraps(original_forward)
    def webgpu_rms_norm_forward(self: Any, hidden_states: Any) -> Any:
        if hidden_states.device.type != "webgpu":
            return original_forward(self, hidden_states)
        weight = self.weight
        if (
            weight.device.type == "webgpu"
            and hidden_states.dtype == torch.float32
            and weight.dtype == torch.float32
        ):
            return torch.nn.functional.rms_norm(
                hidden_states,
                tuple(weight.shape),
                weight,
                self.variance_epsilon,
            )
        return original_forward(self, hidden_states)

    setattr(webgpu_rms_norm_forward, _RMS_NORM_MARKER, target_name)
    return webgpu_rms_norm_forward


def enable_webgpu_rms_norm_fusion() -> dict[str, object]:
    """Fuse the pinned Transformers decoder RMSNorm classes on WebGPU.

    This is an explicit process-wide opt-in for Transformers 4.46.3. It adapts
    only ``Qwen2RMSNorm``, ``LlamaRMSNorm``, ``MistralRMSNorm``, and
    ``Phi3RMSNorm`` from their canonical modules. Float32 WebGPU calls use
    ``torch.nn.functional.rms_norm``; CPU, other-device, and other-dtype calls
    execute the original upstream method unchanged.

    The returned dictionary is JSON-serializable and reports whether each
    target was patched or was already enabled. Repeated calls are safe. A
    version, import, class, or signature mismatch raises
    :class:`TransformersBrowserProfileError` before any target is modified.
    """

    try:
        import torch
        import transformers
    except ImportError as error:
        raise TransformersBrowserProfileError(
            "WebGPU RMSNorm fusion requires torch and Transformers to be "
            "installed before it is enabled; no classes were changed"
        ) from error

    installed_version = getattr(transformers, "__version__", "<unknown>")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise TransformersBrowserProfileError(
            "WebGPU RMSNorm fusion supports exactly Transformers "
            f"{_SUPPORTED_TRANSFORMERS_VERSION}; found {installed_version}. "
            "No classes were changed. Review new Transformers RMSNorm "
            "implementations before extending the adapter."
        )
    functional = getattr(getattr(torch, "nn", None), "functional", None)
    if functional is None or not callable(getattr(functional, "rms_norm", None)):
        raise TransformersBrowserProfileError(
            "WebGPU RMSNorm fusion requires torch.nn.functional.rms_norm; "
            "no classes were changed"
        )

    resolved: list[tuple[type[Any], Callable[[Any, Any], Any], str, bool]] = []
    for module_name, class_name in _RMS_NORM_TARGETS:
        target_name = _rms_norm_target_name(module_name, class_name)
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise TransformersBrowserProfileError(
                f"WebGPU RMSNorm fusion could not import {module_name}: "
                f"{error}. No classes were changed."
            ) from error
        target_class = getattr(module, class_name, None)
        if not isinstance(target_class, type):
            raise TransformersBrowserProfileError(
                f"WebGPU RMSNorm fusion expected class {target_name}; "
                "no classes were changed"
            )
        if (
            target_class.__module__ != module_name
            or target_class.__name__ != class_name
        ):
            raise TransformersBrowserProfileError(
                f"WebGPU RMSNorm fusion found an unexpected class exported as "
                f"{target_name}; no classes were changed"
            )
        forward = getattr(target_class, "forward", None)
        marker = getattr(forward, _RMS_NORM_MARKER, None)
        if marker is not None and marker != target_name:
            raise TransformersBrowserProfileError(
                f"WebGPU RMSNorm fusion found an incompatible adapter on "
                f"{target_name}; no classes were changed"
            )
        already_enabled = marker == target_name
        if not already_enabled:
            if not inspect.isfunction(forward):
                raise TransformersBrowserProfileError(
                    f"WebGPU RMSNorm fusion expected a Python forward method "
                    f"on {target_name}; no classes were changed"
                )
            parameter_names = tuple(inspect.signature(forward).parameters)
            if (
                forward.__module__ != module_name
                or forward.__qualname__ != f"{class_name}.forward"
                or parameter_names != ("self", "hidden_states")
            ):
                raise TransformersBrowserProfileError(
                    f"WebGPU RMSNorm fusion found an unexpected forward "
                    f"implementation on {target_name}; no classes were changed"
                )
        resolved.append((target_class, forward, target_name, already_enabled))

    targets: list[dict[str, str]] = []
    newly_patched = 0
    already_patched = 0
    for target_class, forward, target_name, already_enabled in resolved:
        if already_enabled:
            status = "already_enabled"
            already_patched += 1
        else:
            target_class.forward = _make_webgpu_rms_norm_forward(  # type: ignore[method-assign]
                forward,
                target_name,
                torch,
            )
            status = "patched"
            newly_patched += 1
        targets.append({"target": target_name, "status": status})

    return {
        "enabled": True,
        "profile": "transformers-4.46.3-webgpu-rms-norm",
        "transformers_version": installed_version,
        "supported_transformers_version": _SUPPORTED_TRANSFORMERS_VERSION,
        "newly_patched": newly_patched,
        "already_patched": already_patched,
        "targets": targets,
    }


def _make_webgpu_opt_sdpa_mask_update(
    original_update: Callable[..., Any],
    target_name: str,
    torch: ModuleType,
) -> Callable[..., Any]:
    @wraps(original_update)
    def webgpu_opt_sdpa_mask_update(
        self: Any,
        inputs_embeds: Any,
        input_shape: Any,
        past_key_values_length: int,
        attention_mask: Any = None,
        head_mask: Any = None,
        output_attentions: Any = None,
    ) -> Any:
        try:
            is_webgpu = inputs_embeds.device.type == "webgpu"
            batch_size, sequence_length = input_shape
            can_elide = sequence_length == 1 or past_key_values_length == 0
            eligible = (
                is_webgpu
                and attention_mask is None
                and bool(self._use_sdpa)
                and not bool(self._use_flash_attention_2)
                and not output_attentions
                and head_mask is None
                and can_elide
            )
        except (AttributeError, TypeError, ValueError, RuntimeError):
            eligible = False

        if eligible:
            mask_sequence_length = past_key_values_length + sequence_length
            generated_attention_mask = torch.ones(
                batch_size,
                mask_sequence_length,
                device=inputs_embeds.device,
            )
            return None, generated_attention_mask
        return original_update(
            self,
            inputs_embeds,
            input_shape,
            past_key_values_length,
            attention_mask,
            head_mask,
            output_attentions,
        )

    setattr(
        webgpu_opt_sdpa_mask_update,
        _OPT_SDPA_MASK_MARKER,
        target_name,
    )
    return webgpu_opt_sdpa_mask_update


def enable_webgpu_opt_sdpa_mask_compatibility() -> dict[str, object]:
    """Elide OPT's known all-one SDPA mask check on WebGPU.

    The pinned ``OPTDecoder`` creates an all-one attention mask when callers
    omit one, then synchronously evaluates ``torch.all(mask == 1)`` to decide
    whether SDPA can use ``is_causal``. For an uncached full forward, or a
    one-token cached forward, that answer is known from shapes without reading
    GPU data. This adapter takes only that exact branch and returns the same
    generated mask plus ``None`` for the causal mask. CPU calls, user-provided
    masks, non-SDPA attention, attention outputs, head masks, and multi-token
    cached calls execute the captured upstream method unchanged.

    The exact Transformers version, canonical class, method name, and method
    signature are validated before the process-wide change. Repeated calls are
    idempotent.
    """

    try:
        import torch
        import transformers
    except ImportError as error:
        raise TransformersBrowserProfileError(
            "WebGPU OPT SDPA mask compatibility requires torch and "
            "Transformers to be installed before it is enabled; no classes "
            "were changed"
        ) from error

    installed_version = getattr(transformers, "__version__", "<unknown>")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise TransformersBrowserProfileError(
            "WebGPU OPT SDPA mask compatibility supports exactly "
            f"Transformers {_SUPPORTED_TRANSFORMERS_VERSION}; found "
            f"{installed_version}. No classes were changed. Review OPT mask "
            "construction before extending the adapter."
        )
    if not callable(getattr(torch, "ones", None)):
        raise TransformersBrowserProfileError(
            "WebGPU OPT SDPA mask compatibility requires torch.ones; no "
            "classes were changed"
        )

    module_name, class_name = _OPT_SDPA_MASK_TARGET
    target_name = _rms_norm_target_name(module_name, class_name)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise TransformersBrowserProfileError(
            "WebGPU OPT SDPA mask compatibility could not import "
            f"{module_name}: {error}. No classes were changed."
        ) from error
    target_class = getattr(module, class_name, None)
    if not isinstance(target_class, type):
        raise TransformersBrowserProfileError(
            "WebGPU OPT SDPA mask compatibility expected class "
            f"{target_name}; no classes were changed"
        )
    if (
        target_class.__module__ != module_name
        or target_class.__name__ != class_name
    ):
        raise TransformersBrowserProfileError(
            "WebGPU OPT SDPA mask compatibility found an unexpected class "
            f"exported as {target_name}; no classes were changed"
        )

    update = getattr(target_class, "_update_causal_mask", None)
    marker = getattr(update, _OPT_SDPA_MASK_MARKER, None)
    if marker is not None and marker != target_name:
        raise TransformersBrowserProfileError(
            "WebGPU OPT SDPA mask compatibility found an incompatible "
            f"adapter on {target_name}; no classes were changed"
        )
    already_enabled = marker == target_name
    if not already_enabled:
        if not inspect.isfunction(update):
            raise TransformersBrowserProfileError(
                "WebGPU OPT SDPA mask compatibility expected a Python "
                f"_update_causal_mask method on {target_name}; no classes "
                "were changed"
            )
        parameter_names = tuple(inspect.signature(update).parameters)
        expected_parameters = (
            "self",
            "inputs_embeds",
            "input_shape",
            "past_key_values_length",
            "attention_mask",
            "head_mask",
            "output_attentions",
        )
        if (
            update.__module__ != module_name
            or update.__qualname__ != f"{class_name}._update_causal_mask"
            or parameter_names != expected_parameters
        ):
            raise TransformersBrowserProfileError(
                "WebGPU OPT SDPA mask compatibility found an unexpected "
                f"_update_causal_mask implementation on {target_name}; no "
                "classes were changed"
            )

    status = "already_enabled" if already_enabled else "patched"
    if not already_enabled:
        target_class._update_causal_mask = (  # type: ignore[method-assign]
            _make_webgpu_opt_sdpa_mask_update(
                update,
                target_name,
                torch,
            )
        )

    return {
        "enabled": True,
        "profile": "transformers-4.46.3-webgpu-opt-sdpa-mask",
        "transformers_version": installed_version,
        "supported_transformers_version": _SUPPORTED_TRANSFORMERS_VERSION,
        "newly_patched": 0 if already_enabled else 1,
        "already_patched": 1 if already_enabled else 0,
        "host_mask_truth_readbacks_avoided_per_eligible_call": 1,
        "targets": [{"target": target_name, "status": status}],
    }


def _has_swiglu_bypassed_hooks(module: Any, torch: ModuleType) -> bool:
    """Return whether fusion would skip observable child-module hooks."""

    hook_names = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    for child_name in ("gate_proj", "up_proj", "act_fn"):
        child = getattr(module, child_name, None)
        if child is None:
            return True
        if any(bool(getattr(child, name, None)) for name in hook_names):
            return True

    modules = getattr(getattr(torch, "nn", None), "modules", None)
    module_globals = getattr(modules, "module", None)
    global_hook_names = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
    )
    return module_globals is not None and any(
        bool(getattr(module_globals, name, None)) for name in global_hook_names
    )


def _contiguous_float32_webgpu(tensor: Any, torch: ModuleType) -> bool:
    try:
        return (
            tensor.device.type == "webgpu"
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
        )
    except (AttributeError, TypeError, RuntimeError):
        return False


def _webgpu_swiglu_eligible(module: Any, hidden_states: Any, torch: ModuleType) -> bool:
    """Check the complete narrow contract before bypassing upstream modules."""

    if not _contiguous_float32_webgpu(hidden_states, torch):
        return False
    try:
        linear_type = torch.nn.Linear
        if (
            type(module.gate_proj) is not linear_type
            or type(module.up_proj) is not linear_type
        ):
            return False
        shape = tuple(hidden_states.shape)
        if (
            not shape
            or shape[-1] <= 0
            or hidden_states.numel() != shape[-1]
        ):
            return False
        gate_weight = module.gate_proj.weight
        up_weight = module.up_proj.weight
        if not (
            _contiguous_float32_webgpu(gate_weight, torch)
            and _contiguous_float32_webgpu(up_weight, torch)
            and tuple(gate_weight.shape) == tuple(up_weight.shape)
            and len(gate_weight.shape) == 2
            and gate_weight.shape[0] > 0
            and gate_weight.shape[1] == shape[-1]
        ):
            return False
        for projection in (module.gate_proj, module.up_proj):
            bias = projection.bias
            if bias is not None and not (
                _contiguous_float32_webgpu(bias, torch)
                and tuple(bias.shape) == (gate_weight.shape[0],)
            ):
                return False
        silu_type = getattr(getattr(torch, "nn", None), "SiLU", None)
        if silu_type is None or type(module.act_fn) is not silu_type:
            return False
        if bool(getattr(module.act_fn, "inplace", False)):
            return False
        if getattr(getattr(module, "config", None), "pretraining_tp", 1) != 1:
            return False
        if _has_swiglu_bypassed_hooks(module, torch):
            return False
        is_grad_enabled = getattr(torch, "is_grad_enabled", None)
        if callable(is_grad_enabled) and is_grad_enabled():
            tensors = (
                hidden_states,
                gate_weight,
                up_weight,
                module.gate_proj.bias,
                module.up_proj.bias,
            )
            if any(
                tensor is not None and bool(getattr(tensor, "requires_grad", False))
                for tensor in tensors
            ):
                return False
    except (AttributeError, IndexError, TypeError, RuntimeError):
        return False
    return True


def _make_webgpu_swiglu_forward(
    original_forward: Callable[[Any, Any], Any],
    target_name: str,
    torch: ModuleType,
) -> Callable[[Any, Any], Any]:
    @wraps(original_forward)
    def webgpu_swiglu_forward(self: Any, hidden_states: Any) -> Any:
        if not _webgpu_swiglu_eligible(self, hidden_states, torch):
            return original_forward(self, hidden_states)
        intermediate = torch.ops.webgpu.fused_swiglu(
            hidden_states,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.gate_proj.bias,
            self.up_proj.bias,
        )
        return self.down_proj(intermediate)

    setattr(webgpu_swiglu_forward, _SWIGLU_MARKER, target_name)
    return webgpu_swiglu_forward


def enable_webgpu_swiglu_fusion() -> dict[str, object]:
    """Fuse one-row SwiGLU in pinned Qwen2/Llama/Mistral MLPs on WebGPU.

    The adapter is intentionally decode-specific: only contiguous float32
    WebGPU tensors containing exactly one flattened row take the custom op.
    It also requires stock ``torch.nn.Linear`` gate/up projections, the stock
    non-inplace ``torch.nn.SiLU`` activation, compatible projection parameters,
    no child/global hooks that fusion would bypass, and Llama
    ``pretraining_tp == 1``. Every other call executes the captured
    Transformers 4.46.3 method unchanged.

    The complete target set and custom operator are validated before any class
    is modified. Repeated calls are idempotent.
    """

    try:
        import torch
        import transformers
    except ImportError as error:
        raise TransformersBrowserProfileError(
            "WebGPU SwiGLU fusion requires torch and Transformers to be "
            "installed before it is enabled; no classes were changed"
        ) from error

    installed_version = getattr(transformers, "__version__", "<unknown>")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise TransformersBrowserProfileError(
            "WebGPU SwiGLU fusion supports exactly Transformers "
            f"{_SUPPORTED_TRANSFORMERS_VERSION}; found {installed_version}. "
            "No classes were changed. Review new Transformers MLP "
            "implementations before extending the adapter."
        )
    fused_swiglu = getattr(
        getattr(getattr(torch, "ops", None), "webgpu", None),
        "fused_swiglu",
        None,
    )
    if not callable(fused_swiglu):
        raise TransformersBrowserProfileError(
            "WebGPU SwiGLU fusion requires torch.ops.webgpu.fused_swiglu; "
            "no classes were changed"
        )
    if not isinstance(getattr(getattr(torch, "nn", None), "SiLU", None), type):
        raise TransformersBrowserProfileError(
            "WebGPU SwiGLU fusion requires torch.nn.SiLU; no classes were "
            "changed"
        )
    if not isinstance(getattr(getattr(torch, "nn", None), "Linear", None), type):
        raise TransformersBrowserProfileError(
            "WebGPU SwiGLU fusion requires torch.nn.Linear; no classes were "
            "changed"
        )

    resolved: list[tuple[type[Any], Callable[[Any, Any], Any], str, bool]] = []
    for module_name, class_name, input_name in _SWIGLU_TARGETS:
        target_name = _rms_norm_target_name(module_name, class_name)
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise TransformersBrowserProfileError(
                f"WebGPU SwiGLU fusion could not import {module_name}: "
                f"{error}. No classes were changed."
            ) from error
        target_class = getattr(module, class_name, None)
        if not isinstance(target_class, type):
            raise TransformersBrowserProfileError(
                f"WebGPU SwiGLU fusion expected class {target_name}; no "
                "classes were changed"
            )
        if (
            target_class.__module__ != module_name
            or target_class.__name__ != class_name
        ):
            raise TransformersBrowserProfileError(
                "WebGPU SwiGLU fusion found an unexpected class exported as "
                f"{target_name}; no classes were changed"
            )
        forward = getattr(target_class, "forward", None)
        marker = getattr(forward, _SWIGLU_MARKER, None)
        if marker is not None and marker != target_name:
            raise TransformersBrowserProfileError(
                f"WebGPU SwiGLU fusion found an incompatible adapter on "
                f"{target_name}; no classes were changed"
            )
        already_enabled = marker == target_name
        if not already_enabled:
            if not inspect.isfunction(forward):
                raise TransformersBrowserProfileError(
                    "WebGPU SwiGLU fusion expected a Python forward method on "
                    f"{target_name}; no classes were changed"
                )
            parameter_names = tuple(inspect.signature(forward).parameters)
            if (
                forward.__module__ != module_name
                or forward.__qualname__ != f"{class_name}.forward"
                or parameter_names != ("self", input_name)
            ):
                raise TransformersBrowserProfileError(
                    "WebGPU SwiGLU fusion found an unexpected forward "
                    f"implementation on {target_name}; no classes were changed"
                )
        resolved.append((target_class, forward, target_name, already_enabled))

    targets: list[dict[str, str]] = []
    newly_patched = 0
    already_patched = 0
    for target_class, forward, target_name, already_enabled in resolved:
        if already_enabled:
            status = "already_enabled"
            already_patched += 1
        else:
            target_class.forward = _make_webgpu_swiglu_forward(  # type: ignore[method-assign]
                forward,
                target_name,
                torch,
            )
            status = "patched"
            newly_patched += 1
        targets.append({"target": target_name, "status": status})

    return {
        "enabled": True,
        "profile": "transformers-4.46.3-webgpu-decode-swiglu",
        "transformers_version": installed_version,
        "supported_transformers_version": _SUPPORTED_TRANSFORMERS_VERSION,
        "newly_patched": newly_patched,
        "already_patched": already_patched,
        "upstream_dispatches_before_down_proj": 4,
        "fused_dispatches_before_down_proj": 1,
        "dispatches_saved_per_fused_call": 3,
        "targets": targets,
    }


def _checked_attention_scaling(value: object, target_name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise TransformersBrowserProfileError(
            f"{target_name}.attention_scaling must be a finite Python number "
            "for the pinned WebGPU rotary adapter"
        )
    scaling = float(value)
    if not math.isfinite(scaling):
        raise TransformersBrowserProfileError(
            f"{target_name}.attention_scaling must be finite for the pinned "
            "WebGPU rotary adapter"
        )
    return scaling


def _apply_webgpu_rotary_scaling(
    cos: Any,
    sin: Any,
    scaling: float,
    torch: ModuleType,
) -> tuple[Any, Any]:
    # The default, linear, dynamic, and Llama-3 profiles all return 1.0 in
    # Transformers 4.46.3. Avoiding these two identity operations is exact and
    # also makes the adapter useful with WebGPU wheels predating mul.Scalar.
    if scaling == 1.0:
        return cos, sin
    scalar_mul = torch.ops.aten.mul.Scalar
    return scalar_mul(cos, scaling), scalar_mul(sin, scaling)


def _webgpu_rotary_embedding_forward(
    self: Any,
    x: Any,
    position_ids: Any,
    target_name: str,
    torch: ModuleType,
) -> tuple[Any, Any]:
    """Exact 4.46.3 rotary forward with device-safe attention scaling."""

    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
        position_ids.shape[0], -1, 1
    )
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type
    device_type = (
        device_type
        if isinstance(device_type, str) and device_type != "mps"
        else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (
            inv_freq_expanded.float() @ position_ids_expanded.float()
        ).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

    scaling = _checked_attention_scaling(
        self.attention_scaling,
        target_name,
    )
    cos, sin = _apply_webgpu_rotary_scaling(cos, sin, scaling, torch)
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _make_webgpu_rotary_forward(
    original_forward: Callable[[Any, Any, Any], Any],
    target_name: str,
    torch: ModuleType,
) -> Callable[[Any, Any, Any], Any]:
    @wraps(original_forward)
    @torch.no_grad()
    def webgpu_rotary_forward(
        self: Any,
        x: Any,
        position_ids: Any,
    ) -> Any:
        if x.device.type != "webgpu":
            return original_forward(self, x, position_ids)
        return _webgpu_rotary_embedding_forward(
            self,
            x,
            position_ids,
            target_name,
            torch,
        )

    setattr(webgpu_rotary_forward, _ROTARY_SCALING_MARKER, target_name)
    return webgpu_rotary_forward


def enable_webgpu_rotary_scaling_compatibility() -> dict[str, object]:
    """Adapt pinned Qwen2/Llama rotary attention scaling for WebGPU.

    Transformers 4.46.3 expresses both post-RoPE scales with ``Tensor *
    Python-number``. Its Python binding reaches ``aten::mul.Tensor`` with a
    freshly lifted CPU zero-dimensional tensor. This explicit adapter keeps
    the captured upstream method for every non-WebGPU call, skips the exact
    identity scale, and uses ``aten::mul.Scalar`` for non-identity scaling.

    Only the canonical ``Qwen2RotaryEmbedding`` and ``LlamaRotaryEmbedding``
    classes from exactly Transformers 4.46.3 are accepted. The complete target
    set is validated before either class is changed, and repeated calls are
    idempotent.
    """

    try:
        import torch
        import transformers
    except ImportError as error:
        raise TransformersBrowserProfileError(
            "WebGPU rotary scaling compatibility requires torch and "
            "Transformers to be installed before it is enabled; no classes "
            "were changed"
        ) from error

    installed_version = getattr(transformers, "__version__", "<unknown>")
    if installed_version != _SUPPORTED_TRANSFORMERS_VERSION:
        raise TransformersBrowserProfileError(
            "WebGPU rotary scaling compatibility supports exactly "
            f"Transformers {_SUPPORTED_TRANSFORMERS_VERSION}; found "
            f"{installed_version}. No classes were changed. Review new "
            "Transformers rotary implementations before extending the adapter."
        )
    scalar_mul = getattr(
        getattr(getattr(getattr(torch, "ops", None), "aten", None), "mul", None),
        "Scalar",
        None,
    )
    if not callable(scalar_mul):
        raise TransformersBrowserProfileError(
            "WebGPU rotary scaling compatibility requires "
            "torch.ops.aten.mul.Scalar; no classes were changed"
        )
    if not callable(getattr(torch, "no_grad", None)):
        raise TransformersBrowserProfileError(
            "WebGPU rotary scaling compatibility requires torch.no_grad; "
            "no classes were changed"
        )

    resolved: list[
        tuple[type[Any], Callable[[Any, Any, Any], Any], str, bool]
    ] = []
    for module_name, class_name in _ROTARY_SCALING_TARGETS:
        target_name = _rms_norm_target_name(module_name, class_name)
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise TransformersBrowserProfileError(
                "WebGPU rotary scaling compatibility could not import "
                f"{module_name}: {error}. No classes were changed."
            ) from error
        target_class = getattr(module, class_name, None)
        if not isinstance(target_class, type):
            raise TransformersBrowserProfileError(
                "WebGPU rotary scaling compatibility expected class "
                f"{target_name}; no classes were changed"
            )
        if (
            target_class.__module__ != module_name
            or target_class.__name__ != class_name
        ):
            raise TransformersBrowserProfileError(
                "WebGPU rotary scaling compatibility found an unexpected "
                f"class exported as {target_name}; no classes were changed"
            )
        forward = getattr(target_class, "forward", None)
        marker = getattr(forward, _ROTARY_SCALING_MARKER, None)
        if marker is not None and marker != target_name:
            raise TransformersBrowserProfileError(
                "WebGPU rotary scaling compatibility found an incompatible "
                f"adapter on {target_name}; no classes were changed"
            )
        already_enabled = marker == target_name
        if not already_enabled:
            if not inspect.isfunction(forward):
                raise TransformersBrowserProfileError(
                    "WebGPU rotary scaling compatibility expected a Python "
                    f"forward method on {target_name}; no classes were changed"
                )
            parameter_names = tuple(inspect.signature(forward).parameters)
            if (
                forward.__module__ != module_name
                or forward.__qualname__ != f"{class_name}.forward"
                or parameter_names != ("self", "x", "position_ids")
            ):
                raise TransformersBrowserProfileError(
                    "WebGPU rotary scaling compatibility found an unexpected "
                    f"forward implementation on {target_name}; no classes "
                    "were changed"
                )
        resolved.append((target_class, forward, target_name, already_enabled))

    targets: list[dict[str, str]] = []
    newly_patched = 0
    already_patched = 0
    for target_class, forward, target_name, already_enabled in resolved:
        if already_enabled:
            status = "already_enabled"
            already_patched += 1
        else:
            target_class.forward = _make_webgpu_rotary_forward(  # type: ignore[method-assign]
                forward,
                target_name,
                torch,
            )
            status = "patched"
            newly_patched += 1
        targets.append({"target": target_name, "status": status})

    return {
        "enabled": True,
        "profile": "transformers-4.46.3-webgpu-rotary-scaling",
        "transformers_version": installed_version,
        "supported_transformers_version": _SUPPORTED_TRANSFORMERS_VERSION,
        "newly_patched": newly_patched,
        "already_patched": already_patched,
        "identity_dispatches_saved_per_call": 2,
        "targets": targets,
    }
