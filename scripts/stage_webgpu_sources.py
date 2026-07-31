#!/usr/bin/env python3
"""Stage pinned, network-free WebGPU sources into a PyTorch checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config


ROOT = Path(__file__).resolve().parents[1]


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

    sources = {
        ROOT / "vendor" / "torch-webgpu":
            source / "third_party" / "torch-webgpu",
        ROOT / "vendor" / "emdawnwebgpu":
            source / "third_party" / "emdawnwebgpu",
    }
    for source_tree, destination in sources.items():
        if not source_tree.is_dir():
            raise SystemExit(f"vendored source tree is missing: {source_tree}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_tree, destination)

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
        shutil.copyfile(source_license, wheel_licenses / filename)

    required = [
        source / "third_party" / "torch-webgpu" / "csrc" / "ops" / "binary.cpp",
        source / "third_party" / "torch-webgpu" / "csrc" / "ops" / "unary.cpp",
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
