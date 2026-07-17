#!/usr/bin/env python3
"""Repair the toolchain file omitted from the pyodide-build 0.24.1 wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor" / "pyodide-build-0.24.1" / "Emscripten.cmake"
EXPECTED_SHA256 = "d6e4af93ee3718837c2bd945001389eb980dd51753ac7f0f05e00b74515c02c9"
RELATIVE_DESTINATION = Path("tools/cmake/Modules/Platform/Emscripten.cmake")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_package_root() -> Path:
    spec = importlib.util.find_spec("pyodide_build")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("pyodide_build is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def install_toolchain(destination: Path) -> bool:
    source_digest = digest(SOURCE)
    if source_digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"vendored Pyodide toolchain has SHA-256 {source_digest}, "
            f"expected {EXPECTED_SHA256}"
        )
    if destination.exists():
        destination_digest = digest(destination)
        if destination_digest != EXPECTED_SHA256:
            raise RuntimeError(
                f"existing Pyodide toolchain {destination} does not match "
                f"the pinned file ({destination_digest})"
            )
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, destination)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    package_root = args.package_root or installed_package_root()
    destination = package_root / RELATIVE_DESTINATION
    changed = install_toolchain(destination)
    action = "installed" if changed else "verified"
    print(f"{action}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
