#!/usr/bin/env python3
"""Read and validate the single build manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "build.toml"
CONSTRAINTS_PATH = ROOT / "config" / "build-constraints.txt"

ENV_KEYS = {
    "PYTORCH_REPOSITORY": ("pytorch", "repository"),
    "PYTORCH_REF": ("pytorch", "ref"),
    "PYTORCH_VERSION": ("pytorch", "version"),
    "PYODIDE_VERSION": ("pyodide", "version"),
    "PYODIDE_BUILD_VERSION": ("pyodide", "build_version"),
    "PYTHON_VERSION": ("pyodide", "python_version"),
    "PYTHON_TAG": ("pyodide", "python_tag"),
    "EMSCRIPTEN_VERSION": ("pyodide", "emscripten_version"),
    "MAX_JOBS": ("build", "max_jobs"),
    "MAXIMUM_WHEEL_MIB": ("build", "maximum_wheel_mib"),
    "WHEEL_VERSION": ("host_tools", "wheel_version"),
    "NINJA_VERSION": ("host_tools", "ninja_version"),
    "CMAKE_VERSION": ("host_tools", "cmake_version"),
}


def load() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


def lookup(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for part in path:
        value = value[part]
    return value


def validate(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        ref = str(lookup(config, ("pytorch", "ref")))
        repository = str(lookup(config, ("pytorch", "repository")))
        version = str(lookup(config, ("pytorch", "version")))
        pyodide_version = str(lookup(config, ("pyodide", "version")))
        build_version = str(lookup(config, ("pyodide", "build_version")))
        python_version = str(lookup(config, ("pyodide", "python_version")))
        python_tag = str(lookup(config, ("pyodide", "python_tag")))
        emscripten = str(lookup(config, ("pyodide", "emscripten_version")))
        max_jobs = int(lookup(config, ("build", "max_jobs")))
        max_wheel = int(lookup(config, ("build", "maximum_wheel_mib")))
        wheel_version = str(lookup(config, ("host_tools", "wheel_version")))
        ninja_version = str(lookup(config, ("host_tools", "ninja_version")))
        cmake_version = str(lookup(config, ("host_tools", "cmake_version")))
    except (KeyError, TypeError, ValueError) as exc:
        return [f"missing or invalid manifest value: {exc}"]

    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        errors.append("pytorch.ref must be a full lowercase 40-character commit")
    if repository != "https://github.com/pytorch/pytorch.git":
        errors.append("pytorch.repository must use the canonical HTTPS URL")
    if not re.fullmatch(r"[0-9A-Za-z.+!-]+", version):
        errors.append("pytorch.version contains unexpected characters")
    if pyodide_version != build_version:
        errors.append("pyodide.version and pyodide.build_version must match")
    if not re.fullmatch(r"\d+\.\d+\.\d+", python_version):
        errors.append("pyodide.python_version must be a three-part version")
    else:
        major, minor, _ = python_version.split(".")
        expected_tag = f"cp{major}{minor}"
        if python_tag != expected_tag:
            errors.append(f"python_tag must be {expected_tag} for {python_version}")
    if not re.fullmatch(r"\d+\.\d+\.\d+", emscripten):
        errors.append("pyodide.emscripten_version must be a three-part version")
    if not 1 <= max_jobs <= 16:
        errors.append("build.max_jobs must be between 1 and 16")
    if not 20 <= max_wheel <= 500:
        errors.append("build.maximum_wheel_mib must be between 20 and 500")
    for name, value in (
        ("host_tools.wheel_version", wheel_version),
        ("host_tools.ninja_version", ninja_version),
        ("host_tools.cmake_version", cmake_version),
    ):
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:\.\d+)?", value):
            errors.append(f"{name} must be an exact numeric version")
    expected_constraints = [
        f"cmake=={cmake_version}",
        f"ninja=={ninja_version}",
        f"wheel=={wheel_version}",
    ]
    try:
        constraints = [
            line.strip()
            for line in CONSTRAINTS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    except OSError as exc:
        errors.append(f"cannot read build constraints: {exc}")
    else:
        if constraints != expected_constraints:
            errors.append(
                "config/build-constraints.txt must match host_tools exactly"
            )
    return errors


def flat_env(config: dict[str, Any]) -> dict[str, str]:
    return {name: str(lookup(config, path)) for name, path in ENV_KEYS.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    subparsers.add_parser("env")
    subparsers.add_parser("github-output")
    subparsers.add_parser("json")
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("key", choices=sorted(ENV_KEYS))
    args = parser.parse_args()

    config = load()
    errors = validate(config)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    values = flat_env(config)
    if args.command == "check":
        print(f"valid: {CONFIG_PATH}")
    elif args.command in {"env", "github-output"}:
        for name, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"{name} contains a newline")
            key = name if args.command == "env" else name.lower()
            print(f"{key}={value}")
    elif args.command == "json":
        print(json.dumps(config, indent=2, sort_keys=True))
    elif args.command == "get":
        print(values[args.key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
