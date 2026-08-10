#!/usr/bin/env python3
"""Stage pinned, network-free WebGPU sources into a PyTorch checkout."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

import config


ROOT = Path(__file__).resolve().parents[1]


SHADERS = {
    "all_bool.wgsl": "kAllBool",
    "any_bool.wgsl": "kAnyBool",
    "arange.wgsl": "kArange",
    "argmax.wgsl": "kArgmax",
    "bitwise_not_bool.wgsl": "kBitwiseNotBool",
    "bool_to_long.wgsl": "kBoolToLong",
    "baddbmm.wgsl": "kBaddbmm",
    "bmm.wgsl": "kBmm",
    "embedding.wgsl": "kEmbedding",
    "eq_scalar.wgsl": "kEqScalar",
    "fill.wgsl": "kFill",
    "gemma_rms_norm.wgsl": "kGemmaRmsNorm",
    "gt_tensor.wgsl": "kGtTensor",
    "int_to_float.wgsl": "kIntToFloat",
    "kv_cache_update.wgsl": "kKvCacheUpdate",
    "layer_norm.wgsl": "kLayerNorm",
    "linear.wgsl": "kLinear",
    "linear_gemm_16x32.wgsl": "kLinearGemm16x32",
    "linear_gemv.wgsl": "kLinearGemv",
    "linear_gemv_q8_s4.wgsl": "kLinearGemvQ8S4",
    "linear_gemv_subgroup_s4.wgsl": "kLinearGemvSubgroupS4",
    "long_cumsum.wgsl": "kLongCumsum",
    "long_arithmetic.wgsl": "kLongArithmetic",
    "long_isin.wgsl": "kLongIsin",
    "long_lt_scalar.wgsl": "kLongLtScalar",
    "masked_fill_scalar.wgsl": "kMaskedFillScalar",
    "mean_dim.wgsl": "kMeanDim",
    "mixed_pow.wgsl": "kMixedPow",
    "mul_bool_inplace.wgsl": "kMulBoolInplace",
    "mul_bool_tensor.wgsl": "kMulBoolTensor",
    "ne_tensor.wgsl": "kNeTensor",
    "rms_norm.wgsl": "kRmsNorm",
    "scalar_binary.wgsl": "kScalarBinary",
    "sdpa.wgsl": "kSdpa",
    "strided_copy.wgsl": "kStridedCopy",
    "swiglu_gemv.wgsl": "kSwiGluGemv",
    "swiglu_gemv_subgroup_s4.wgsl": "kSwiGluGemvSubgroupS4",
    "triangular.wgsl": "kTriangular",
    "where_float.wgsl": "kWhereFloat",
}


def copy_file_if_changed(source: Path, destination: Path) -> None:
    """Copy a file without touching a byte-identical destination."""
    if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sync_tree(
    source: Path,
    destination: Path,
    *,
    preserve: frozenset[Path] = frozenset(),
) -> None:
    """Mirror a source tree while preserving mtimes for unchanged files."""
    source_entries = {
        path.relative_to(source): path for path in source.rglob("*")
    }
    if destination.exists() and not destination.is_dir():
        destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)

    # Remove inputs that disappeared from the pinned source snapshot. Work
    # deepest-first so stale directories are empty before they are removed.
    for target in sorted(
        destination.rglob("*"),
        key=lambda path: len(path.relative_to(destination).parts),
        reverse=True,
    ):
        relative = target.relative_to(destination)
        source_path = source_entries.get(relative)
        if relative in preserve:
            continue
        wrong_type = source_path is not None and (
            source_path.is_dir() != target.is_dir()
        )
        if source_path is None or wrong_type:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    for relative, source_path in source_entries.items():
        target = destination / relative
        if source_path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            copy_file_if_changed(source_path, target)
        else:
            raise SystemExit(f"unsupported staged source entry: {source_path}")


def write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def embed_project_shaders(directory: Path) -> None:
    lines = [
        "#pragma once",
        "",
        "namespace pyodide_pytorch::webgpu::shaders {",
    ]
    for filename, symbol in SHADERS.items():
        shader = (directory / filename).read_text(encoding="utf-8")
        if ")wgsl\"" in shader:
            raise SystemExit(f"WGSL raw-string delimiter occurs in {filename}")
        lines.extend(
            (
                "",
                f"inline constexpr char {symbol}[] = R\"wgsl({shader})wgsl\";",
            )
        )
    lines.extend(("", "} // namespace pyodide_pytorch::webgpu::shaders", ""))
    write_text_if_changed(directory / "embedded_shaders.h", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pytorch_source", type=Path)
    args = parser.parse_args()

    source = args.pytorch_source.resolve()
    if not (source / "torch" / "CMakeLists.txt").is_file():
        raise SystemExit(f"not a PyTorch checkout: {source}")

    values = config.flat_env(config.load())
    torch_webgpu_ref = (
        ROOT / "vendor" / "torch-webgpu" / "COMMIT"
    ).read_text(encoding="utf-8").strip()
    if torch_webgpu_ref != values["TORCH_WEBGPU_REF"]:
        raise SystemExit("vendored torch-webgpu COMMIT does not match config")
    emdawn_version = (ROOT / "vendor" / "emdawnwebgpu" / "VERSION.txt").read_text(
        encoding="utf-8"
    )
    if values["EMDAWNWEBGPU_RELEASE"] not in emdawn_version:
        raise SystemExit("vendored Emdawnwebgpu VERSION.txt does not match config")
    if values["EMDAWNWEBGPU_DAWN_REF"] not in emdawn_version:
        raise SystemExit("vendored Emdawnwebgpu Dawn revision does not match config")

    sources = (
        (
            ROOT / "vendor" / "torch-webgpu",
            source / "third_party" / "torch-webgpu",
            frozenset(),
        ),
        (
            ROOT / "vendor" / "emdawnwebgpu",
            source / "third_party" / "emdawnwebgpu",
            frozenset(),
        ),
        (
            ROOT / "webgpu" / "llm_kernels",
            source / "third_party" / "pyodide-pytorch-webgpu" / "llm_kernels",
            frozenset({Path("embedded_shaders.h")}),
        ),
    )
    for source_tree, destination, preserve in sources:
        if not source_tree.is_dir():
            raise SystemExit(f"vendored source tree is missing: {source_tree}")
        sync_tree(source_tree, destination, preserve=preserve)

    project_kernels = (
        source / "third_party" / "pyodide-pytorch-webgpu" / "llm_kernels"
    )
    embed_project_shaders(project_kernels)

    wheel_licenses = source / "torch" / "webgpu" / "licenses"
    wheel_licenses.mkdir(parents=True, exist_ok=True)
    for source_license, filename in (
        (
            ROOT / "vendor" / "torch-webgpu" / "LICENSE",
            "torch-webgpu-Apache-2.0.txt",
        ),
        (
            ROOT / "vendor" / "emdawnwebgpu" / "webgpu" / "src" / "LICENSE",
            "emdawnwebgpu-webgpu-license.txt",
        ),
        (
            ROOT / "vendor" / "emdawnwebgpu" / "webgpu_cpp" / "LICENSE",
            "emdawnwebgpu-webgpu-cpp-BSD-3-Clause.txt",
        ),
    ):
        copy_file_if_changed(source_license, wheel_licenses / filename)

    required = [
        source / "third_party" / "torch-webgpu" / "csrc" / "ops" / "binary.cpp",
        source / "third_party" / "torch-webgpu" / "csrc" / "ops" / "unary.cpp",
        project_kernels / "boolean.cpp",
        project_kernels / "browser_unary.cpp",
        project_kernels / "all_bool.wgsl",
        project_kernels / "any_bool.wgsl",
        project_kernels / "bitwise_not_bool.wgsl",
        project_kernels / "bool_to_long.wgsl",
        project_kernels / "creation.cpp",
        project_kernels / "arange.wgsl",
        project_kernels / "argmax.cpp",
        project_kernels / "argmax.wgsl",
        project_kernels / "fill.wgsl",
        project_kernels / "gemma_rms_norm.cpp",
        project_kernels / "gemma_rms_norm.wgsl",
        project_kernels / "gt_tensor.wgsl",
        project_kernels / "generation_long.cpp",
        project_kernels / "generation_control.cpp",
        project_kernels / "generation_control.h",
        project_kernels / "eq_scalar.wgsl",
        project_kernels / "llm_common.h",
        project_kernels / "kv_cache.cpp",
        project_kernels / "kv_cache_update.wgsl",
        project_kernels / "q8_linear.cpp",
        project_kernels / "linear_gemv_q8_s4.wgsl",
        project_kernels / "long_cumsum.wgsl",
        project_kernels / "long_arithmetic.cpp",
        project_kernels / "long_arithmetic.h",
        project_kernels / "long_arithmetic.wgsl",
        project_kernels / "long_isin.wgsl",
        project_kernels / "long_lt_scalar.wgsl",
        project_kernels / "mixed_pow.wgsl",
        project_kernels / "masking.cpp",
        project_kernels / "masked_fill.cpp",
        project_kernels / "masked_fill_scalar.wgsl",
        project_kernels / "mul_bool_inplace.wgsl",
        project_kernels / "mul_bool_tensor.wgsl",
        project_kernels / "ne_tensor.wgsl",
        project_kernels / "normalization.cpp",
        project_kernels / "layer_norm.wgsl",
        project_kernels / "scalar_binary.cpp",
        project_kernels / "scalar_binary.wgsl",
        project_kernels / "swiglu.cpp",
        project_kernels / "swiglu_gemv.wgsl",
        project_kernels / "swiglu_gemv_subgroup_s4.wgsl",
        project_kernels / "triangular.wgsl",
        project_kernels / "embedded_shaders.h",
        project_kernels / "where_float.wgsl",
        source / "third_party" / "emdawnwebgpu" / "webgpu" / "include" / "webgpu" / "webgpu.h",
        source / "third_party" / "emdawnwebgpu" / "webgpu_cpp" / "include" / "webgpu" / "webgpu_cpp.h",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("staged WebGPU sources are incomplete: " + ", ".join(missing))
    print(f"staged pinned WebGPU sources in {source / 'third_party'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
