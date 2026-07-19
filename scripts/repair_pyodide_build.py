#!/usr/bin/env python3
"""Apply compatibility repairs to the pyodide-build 0.24.1 wheel."""

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
PYWASMCROSS_RELATIVE_DESTINATION = Path("pywasmcross.py")
PYWASMCROSS_EXPECTED_SHA256 = (
    "f06343ce8461d0f3121c96179915c096fdd8c953ba2ccb10decf4483b76f6f78"
)
PYWASMCROSS_FIXED_SHA256 = (
    "bd93abfccebdd316969e57b000f50fc749ece799d1cdea885d927876e4c58aa3"
)
CMAKE_COMMAND_MODE_ORIGINAL = (
    '        if "--build" in line or "--install" in line or "-P" in line:'
)
CMAKE_COMMAND_MODE_FIXED = (
    '        if "--build" in line or "--install" in line or "-P" in line '
    'or "-E" in line:'
)


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


def install_cmake_command_mode_fix(destination: Path) -> bool:
    """Backport Pyodide PR 4705 so ``cmake -E`` bypasses cross wrapping."""

    source_digest = digest(destination)
    if source_digest == PYWASMCROSS_FIXED_SHA256:
        return False
    if source_digest != PYWASMCROSS_EXPECTED_SHA256:
        raise RuntimeError(
            f"pywasmcross {destination} has SHA-256 {source_digest}, expected "
            f"{PYWASMCROSS_EXPECTED_SHA256}"
        )

    source = destination.read_text(encoding="utf-8")
    original_count = source.count(CMAKE_COMMAND_MODE_ORIGINAL)
    fixed_count = source.count(CMAKE_COMMAND_MODE_FIXED)
    if original_count != 1 or fixed_count != 0:
        raise RuntimeError(
            f"cannot safely repair CMake command mode in {destination}: "
            f"found original={original_count}, fixed={fixed_count}"
        )
    destination.write_text(
        source.replace(CMAKE_COMMAND_MODE_ORIGINAL, CMAKE_COMMAND_MODE_FIXED),
        encoding="utf-8",
    )
    fixed_digest = digest(destination)
    if fixed_digest != PYWASMCROSS_FIXED_SHA256:
        raise RuntimeError(
            f"repaired pywasmcross {destination} has SHA-256 {fixed_digest}, "
            f"expected {PYWASMCROSS_FIXED_SHA256}"
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    package_root = args.package_root or installed_package_root()
    toolchain_destination = package_root / RELATIVE_DESTINATION
    toolchain_changed = install_toolchain(toolchain_destination)
    toolchain_action = "installed" if toolchain_changed else "verified"
    print(f"{toolchain_action}: {toolchain_destination}")

    pywasmcross_destination = package_root / PYWASMCROSS_RELATIVE_DESTINATION
    wrapper_changed = install_cmake_command_mode_fix(pywasmcross_destination)
    wrapper_action = "installed" if wrapper_changed else "verified"
    print(f"{wrapper_action}: CMake command-mode fix in {pywasmcross_destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
