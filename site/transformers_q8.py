"""Opt-in weight-only Q8 conversion for single-row WebGPU decode.

This module is deliberately separate from the general Transformers bootstrap.
It changes module/state-dict structure and supports only batch-one, token-one
decode. Call ``model.eval()`` and convert on CPU before moving other tensors to
WebGPU. The replacement modules do not retain their original float32 weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


Q8_FORMAT_VERSION = 1
Q8_GROUP_SIZE = 128
Q8_VALUES_PER_WORD = 4


@dataclass(frozen=True)
class Q8ConversionRecord:
    name: str
    in_features: int
    out_features: int
    float_weight_bytes: int
    packed_weight_bytes: int


def pack_q8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a contiguous CPU float32 ``[N, K]`` weight into Q8 version 1.

    Four signed int8 values occupy each int32 word in little-endian byte order.
    ``scales[n, g]`` is the float32 absmax for row ``n`` and 128-value group
    ``g``. Quantized values are clamped to ``[-127, 127]`` so WGSL's
    ``unpack4x8snorm`` reconstructs ``q / 127 * absmax`` symmetrically.
    """

    if weight.device.type != "cpu":
        raise ValueError("Q8 packing must run on CPU before WebGPU upload")
    if weight.dtype != torch.float32:
        raise TypeError("Q8 packing requires a torch.float32 weight")
    if weight.ndim != 2:
        raise ValueError("Q8 packing requires a rank-2 [N, K] weight")
    if not weight.is_contiguous():
        raise ValueError("Q8 packing requires a contiguous weight")
    columns, inner = weight.shape
    if columns <= 0 or inner <= 0 or inner % Q8_GROUP_SIZE != 0:
        raise ValueError("Q8 packing requires N > 0 and K divisible by 128")

    grouped = weight.detach().reshape(columns, inner // Q8_GROUP_SIZE, Q8_GROUP_SIZE)
    scales = grouped.abs().amax(dim=-1).to(torch.float32)
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.round(grouped * (127.0 / safe_scales.unsqueeze(-1)))
    quantized = quantized.clamp(-127, 127).to(torch.int32)
    bytes_ = quantized & 0xFF
    packed = (
        bytes_[..., 0::Q8_VALUES_PER_WORD]
        | (bytes_[..., 1::Q8_VALUES_PER_WORD] << 8)
        | (bytes_[..., 2::Q8_VALUES_PER_WORD] << 16)
        | (bytes_[..., 3::Q8_VALUES_PER_WORD] << 24)
    )
    return (
        packed.reshape(columns, inner // Q8_VALUES_PER_WORD).contiguous(),
        scales.contiguous(),
    )


class WebGPUQ8Linear(nn.Module):
    """A weight-owning, decode-only replacement for ``torch.nn.Linear``."""

    __constants__ = [
        "in_features",
        "out_features",
        "group_size",
        "format_version",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if packed_weight.dtype != torch.int32:
            raise TypeError("packed_weight must be torch.int32")
        if scales.dtype != torch.float32:
            raise TypeError("scales must be torch.float32")
        if tuple(packed_weight.shape) != (
            out_features,
            in_features // Q8_VALUES_PER_WORD,
        ):
            raise ValueError("packed_weight must have shape [N, K/4]")
        if tuple(scales.shape) != (
            out_features,
            in_features // Q8_GROUP_SIZE,
        ):
            raise ValueError("scales must have shape [N, K/128]")
        if bias is not None and (
            bias.dtype != torch.float32 or tuple(bias.shape) != (out_features,)
        ):
            raise ValueError("bias must be a float32 [N] tensor")

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = Q8_GROUP_SIZE
        self.format_version = Q8_FORMAT_VERSION
        self.register_buffer("packed_weight", packed_weight.contiguous())
        self.register_buffer("scales", scales.contiguous())
        self.register_buffer("bias", None if bias is None else bias.contiguous())

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        *,
        device: str | torch.device = "webgpu",
    ) -> "WebGPUQ8Linear":
        if linear.training:
            raise ValueError("Q8 conversion requires eval mode")
        weight = linear.weight.detach()
        packed_cpu, scales_cpu = pack_q8_weight(weight)
        bias_cpu = None
        if linear.bias is not None:
            bias_cpu = linear.bias.detach()
            if bias_cpu.device.type != "cpu" or bias_cpu.dtype != torch.float32:
                raise TypeError("Q8 conversion requires a CPU float32 bias")
            bias_cpu = bias_cpu.contiguous()

        # Only the packed representation crosses to WebGPU. The replacement
        # has no reference to ``linear.weight``, so replacing the source module
        # releases the full-precision weight when no other tie owns it.
        replacement = cls(
            linear.in_features,
            linear.out_features,
            packed_cpu.to(device),
            scales_cpu.to(device),
            None if bias_cpu is None else bias_cpu.to(device),
        )
        replacement.train(False)
        return replacement

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] != self.in_features:
            raise ValueError("Q8 linear input feature mismatch")
        if input.numel() != self.in_features:
            raise RuntimeError(
                "WebGPU Q8 linear is decode-only: expected exactly one "
                "flattened input row (batch=1, sequence=1)"
            )
        return torch.ops.webgpu.q8_linear(
            input,
            self.packed_weight,
            self.scales,
            self.bias,
            self.group_size,
            self.format_version,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, format_version={self.format_version}, "
            f"bias={self.bias is not None}, decode_only=True"
        )


LinearPredicate = Callable[[str, nn.Linear], bool]


def _linear_candidates(
    module: nn.Module,
    predicate: LinearPredicate,
) -> list[tuple[nn.Module, str, str, nn.Linear]]:
    candidates: list[tuple[nn.Module, str, str, nn.Linear]] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and predicate(qualified_name, child):
                candidates.append((parent, child_name, qualified_name, child))
            else:
                visit(child, qualified_name)

    visit(module, "")
    return candidates


def convert_linear_modules_q8_(
    module: nn.Module,
    *,
    device: str | torch.device = "webgpu",
    predicate: LinearPredicate | None = None,
) -> list[Q8ConversionRecord]:
    """Opt in selected ``nn.Linear`` children to decode-only WebGPU Q8.

    The conversion preflights every selected child before mutating ``module``.
    It intentionally has no transparent fp32 fallback: prompt prefill with more
    than one flattened row raises, as do unsupported dimensions or devices.
    Pass a predicate to retain float32 modules that must serve prefill or have
    an incompatible K dimension.
    """

    if module.training:
        raise ValueError("Q8 conversion requires module.eval()")
    select = predicate if predicate is not None else (lambda _name, _linear: True)
    candidates = _linear_candidates(module, select)

    failures: list[str] = []
    for _parent, _child_name, qualified_name, linear in candidates:
        weight = linear.weight
        if weight.device.type != "cpu":
            failures.append(qualified_name + ": weight is not on CPU")
        elif weight.dtype != torch.float32:
            failures.append(qualified_name + ": weight is not float32")
        elif not weight.is_contiguous():
            failures.append(qualified_name + ": weight is not contiguous")
        elif linear.in_features <= 0 or linear.in_features % Q8_GROUP_SIZE != 0:
            failures.append(qualified_name + ": in_features is not divisible by 128")
        elif linear.out_features <= 0:
            failures.append(qualified_name + ": out_features is empty")
        elif linear.bias is not None and (
            linear.bias.device.type != "cpu" or linear.bias.dtype != torch.float32
        ):
            failures.append(qualified_name + ": bias is not CPU float32")
    if failures:
        raise ValueError("Q8 conversion preflight failed:\n" + "\n".join(failures))

    records: list[Q8ConversionRecord] = []
    with torch.no_grad():
        for parent, child_name, qualified_name, linear in candidates:
            replacement = WebGPUQ8Linear.from_float(linear, device=device)
            float_bytes = linear.weight.numel() * linear.weight.element_size()
            packed_bytes = (
                replacement.packed_weight.numel()
                * replacement.packed_weight.element_size()
                + replacement.scales.numel() * replacement.scales.element_size()
            )
            parent._modules[child_name] = replacement
            records.append(
                Q8ConversionRecord(
                    name=qualified_name,
                    in_features=linear.in_features,
                    out_features=linear.out_features,
                    float_weight_bytes=float_bytes,
                    packed_weight_bytes=packed_bytes,
                )
            )
    return records
