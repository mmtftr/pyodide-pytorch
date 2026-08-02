#!/usr/bin/env python3
"""Assemble a validated playground tree from local or promoted artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import config
import prepare_transformers_browser_deps
import verify_release_artifact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_DIR = ROOT / "site"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "transformers_tiny.json"
RELEASE_TAG = re.compile(
    r"torch-[0-9]+\.[0-9]+\.[0-9]+-pyodide-[0-9]+\.[0-9]+\.[0-9]+-r[1-9][0-9]*"
)
SHELL_FILES = (
    "index.html",
    "styles.css",
    "worker.js",
    "service-worker.js",
    "transformers_browser_bootstrap.py",
    "transformers_gemma2_webgpu.py",
    "transformers_q8.py",
)
DIST_FILES = ("app.js", "app.js.map")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def validate_release_artifacts(directory: Path) -> tuple[Path, dict[str, Any]]:
    wheel, manifest, errors = verify_release_artifact.verify_contents(directory)
    if errors:
        raise ValueError("invalid release artifact: " + "; ".join(errors))
    assert wheel is not None and manifest is not None

    configuration = manifest.get("configuration")
    if configuration != config.load():
        raise ValueError(
            "release artifact configuration does not match the current pins"
        )
    release_tag = configuration.get("release", {}).get("tag")
    if not isinstance(release_tag, str) or RELEASE_TAG.fullmatch(release_tag) is None:
        raise ValueError("release artifact has an invalid release tag")
    builder_commit = manifest.get("builder_repository_commit")
    if builder_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", builder_commit):
        raise ValueError("release artifact has an invalid builder commit")
    return wheel, manifest


def validate_transformers_artifacts(directory: Path) -> list[dict[str, object]]:
    requirements = prepare_transformers_browser_deps.parse_requirements(
        prepare_transformers_browser_deps.DEFAULT_REQUIREMENTS
    )
    packages = prepare_transformers_browser_deps.verify_wheels(
        requirements, directory
    )
    manifest_path = directory / prepare_transformers_browser_deps.MANIFEST_NAME
    manifest = load_json(manifest_path, "Transformers browser manifest")
    expected_manifest = {
        "schema_version": 1,
        "requirements": "config/transformers-browser-requirements.txt",
        "model_only": True,
        "tokenizers_included": False,
        "packages": packages,
    }
    if manifest != expected_manifest:
        raise ValueError(
            "Transformers browser manifest does not match the pinned wheel inventory"
        )
    files = sorted(item.name for item in directory.iterdir() if item.is_file())
    expected_files = sorted(
        [prepare_transformers_browser_deps.MANIFEST_NAME]
        + [str(package["filename"]) for package in packages]
    )
    if files != expected_files:
        raise ValueError("Transformers artifact contains unexpected companion files")
    return packages


def require_files(directory: Path, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"{label} omits: {', '.join(missing)}")


def assemble(
    release_dir: Path,
    transformers_dir: Path,
    output_dir: Path,
    *,
    site_dir: Path = DEFAULT_SITE_DIR,
    fixture: Path = DEFAULT_FIXTURE,
) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    transformers_dir = transformers_dir.resolve()
    output_dir = output_dir.resolve()
    site_dir = site_dir.resolve()
    fixture = fixture.resolve()

    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    require_files(site_dir, SHELL_FILES, "playground shell")
    require_files(site_dir / "dist", DIST_FILES, "built playground bundle")
    if not fixture.is_file():
        raise ValueError(f"Transformers fixture does not exist: {fixture}")

    wheel, release_manifest = validate_release_artifacts(release_dir)
    transformer_packages = validate_transformers_artifacts(transformers_dir)
    fixture_data = load_json(fixture, "Transformers fixture")
    transformer_version = next(
        package["version"]
        for package in transformer_packages
        if prepare_transformers_browser_deps.normalize_name(str(package["name"]))
        == "transformers"
    )
    if fixture_data.get("transformers_version") != transformer_version:
        raise ValueError(
            "Transformers fixture version does not match the pinned wheel inventory"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        runtime = staging / "runtime"
        transformer_runtime = runtime / "transformers"
        transformer_runtime.mkdir(parents=True)
        for name in SHELL_FILES:
            shutil.copy2(site_dir / name, staging / name)
        for name in DIST_FILES:
            shutil.copy2(site_dir / "dist" / name, staging / name)
        shutil.copy2(release_dir / "build-manifest.json", runtime)
        shutil.copy2(wheel, runtime)
        shutil.copy2(release_dir / f"{wheel.name}.sha256", runtime)
        for package in transformer_packages:
            shutil.copy2(
                transformers_dir / str(package["filename"]), transformer_runtime
            )
        shutil.copy2(
            transformers_dir / prepare_transformers_browser_deps.MANIFEST_NAME,
            transformer_runtime,
        )
        shutil.copy2(fixture, transformer_runtime)
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "output": str(output_dir),
        "release": release_manifest["configuration"]["release"]["tag"],
        "wheel": wheel.name,
        "transformers": transformer_version,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--transformers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_SITE_DIR)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    try:
        result = assemble(
            args.release_dir,
            args.transformers_dir,
            args.output_dir,
            site_dir=args.site_dir,
            fixture=args.fixture,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
