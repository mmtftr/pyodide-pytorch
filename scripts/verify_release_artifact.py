#!/usr/bin/env python3
"""Verify a downloaded release artifact against the pinned build inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import config


ROOT = Path(__file__).resolve().parents[1]
SHA256_LINE = re.compile(
    r"(?P<digest>[0-9a-f]{64})  (?P<filename>[^/\\\r\n]+)\n?"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_inputs() -> dict[str, Any]:
    patches = sorted((ROOT / "patches" / "pytorch").glob("*.patch"))
    return {
        "build_script_sha256": sha256(ROOT / "scripts" / "build_wheel.sh"),
        "patches": [
            {"filename": patch.name, "sha256": sha256(patch)}
            for patch in patches
        ],
    }


def verify(directory: Path, builder_commit: str) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    wheels = sorted(directory.glob("*.whl"))
    manifests = sorted(directory.glob("build-manifest.json"))
    checksums = sorted(directory.glob("*.sha256"))

    if len(wheels) != 1:
        errors.append(f"expected exactly one wheel, found {len(wheels)}")
    if len(manifests) != 1:
        errors.append(f"expected exactly one build-manifest.json, found {len(manifests)}")
    if len(checksums) != 1:
        errors.append(f"expected exactly one SHA-256 file, found {len(checksums)}")
    if errors:
        return (wheels[0] if len(wheels) == 1 else None), errors

    wheel = wheels[0]
    manifest_path = manifests[0]
    checksum_path = checksums[0]
    if checksum_path.name != f"{wheel.name}.sha256":
        errors.append("checksum filename does not match the wheel filename")

    checksum_match = SHA256_LINE.fullmatch(
        checksum_path.read_text(encoding="utf-8")
    )
    actual_digest = sha256(wheel)
    if checksum_match is None:
        errors.append("checksum must contain one portable sha256sum line")
    else:
        if checksum_match.group("filename") != wheel.name:
            errors.append("checksum line does not name the downloaded wheel")
        if checksum_match.group("digest") != actual_digest:
            errors.append("checksum does not match the downloaded wheel")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read build manifest: {exc}")
        return wheel, errors

    if manifest.get("schema_version") != 1:
        errors.append("unsupported build manifest schema")
    if manifest.get("builder_repository_commit") != builder_commit:
        errors.append("build manifest commit does not match the source workflow run")
    if manifest.get("configuration") != config.load():
        errors.append("build manifest configuration does not match the current pins")
    if manifest.get("inputs") != expected_inputs():
        errors.append("build manifest input hashes do not match the current sources")

    wheel_record = manifest.get("wheel")
    expected_wheel = {
        "filename": wheel.name,
        "sha256": actual_digest,
        "size": wheel.stat().st_size,
    }
    if wheel_record != expected_wheel:
        errors.append("build manifest wheel metadata does not match the downloaded wheel")
    return wheel, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--builder-commit",
        required=True,
        help="head SHA reported by the source GitHub Actions run",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.builder_commit):
        parser.error("--builder-commit must be a full lowercase commit SHA")
    wheel, errors = verify(args.directory, args.builder_commit)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    assert wheel is not None
    print(f"verified release artifact: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
