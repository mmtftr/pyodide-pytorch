#!/usr/bin/env python3
"""Download and verify the hashed, model-only Transformers browser wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "config" / "transformers-browser-requirements.txt"
MANIFEST_NAME = "transformers-browser-manifest.json"
REQUIREMENT_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s]+)\s+"
    r"--hash=sha256:(?P<sha256>[0-9a-f]{64})"
)


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class PinnedRequirement:
    name: str
    version: str
    sha256: str

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.name)


def parse_requirements(path: Path) -> list[PinnedRequirement]:
    logical_lines: list[str] = []
    pending = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        logical_lines.append((pending + line).strip())
        pending = ""
    if pending:
        raise ValueError(f"unterminated requirement continuation in {path}")

    requirements: list[PinnedRequirement] = []
    seen: set[str] = set()
    for line in logical_lines:
        match = REQUIREMENT_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"browser requirement must be an exact pin with one SHA-256 hash: {line}"
            )
        requirement = PinnedRequirement(**match.groupdict())
        if requirement.normalized_name in seen:
            raise ValueError(f"duplicate browser requirement: {requirement.name}")
        if requirement.normalized_name == "tokenizers":
            raise ValueError("compiled tokenizers must not enter the model-only browser profile")
        seen.add(requirement.normalized_name)
        requirements.append(requirement)
    if not requirements:
        raise ValueError(f"no browser requirements found in {path}")
    return requirements


def wheel_metadata(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        wheel_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        if len(metadata_files) != 1 or len(wheel_files) != 1:
            raise ValueError(f"{path.name} has an invalid wheel metadata layout")
        metadata = Parser().parsestr(
            archive.read(metadata_files[0]).decode("utf-8")
        )
        wheel = Parser().parsestr(archive.read(wheel_files[0]).decode("utf-8"))
    if wheel.get("Root-Is-Purelib", "").lower() != "true":
        raise ValueError(f"{path.name} is not a pure-Python wheel")
    tags = wheel.get_all("Tag", [])
    if not tags or any(not tag.endswith("-none-any") for tag in tags):
        raise ValueError(f"{path.name} is not browser-portable: {tags}")
    name = metadata.get("Name")
    version = metadata.get("Version")
    if not name or not version:
        raise ValueError(f"{path.name} omits Name or Version metadata")
    return name, version


def verify_wheels(
    requirements: list[PinnedRequirement], output_dir: Path
) -> list[dict[str, object]]:
    expected = {item.normalized_name: item for item in requirements}
    verified: dict[str, dict[str, object]] = {}
    for wheel_path in sorted(output_dir.glob("*.whl")):
        name, version = wheel_metadata(wheel_path)
        normalized_name = normalize_name(name)
        requirement = expected.get(normalized_name)
        if requirement is None:
            raise ValueError(f"unexpected browser wheel: {wheel_path.name}")
        if normalized_name in verified:
            raise ValueError(f"multiple browser wheels for {requirement.name}")
        if version != requirement.version:
            raise ValueError(
                f"{wheel_path.name} has version {version}, expected {requirement.version}"
            )
        digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        if digest != requirement.sha256:
            raise ValueError(
                f"{wheel_path.name} has SHA-256 {digest}, expected {requirement.sha256}"
            )
        verified[normalized_name] = {
            "name": requirement.name,
            "version": requirement.version,
            "filename": wheel_path.name,
            "sha256": digest,
            "size": wheel_path.stat().st_size,
        }

    missing = [
        requirement.name
        for requirement in requirements
        if requirement.normalized_name not in verified
    ]
    if missing:
        raise ValueError(f"missing browser wheels: {', '.join(missing)}")
    return [verified[requirement.normalized_name] for requirement in requirements]


def write_manifest(packages: list[dict[str, object]], output_dir: Path) -> Path:
    manifest_path = output_dir / MANIFEST_NAME
    manifest = {
        "schema_version": 1,
        "requirements": "config/transformers-browser-requirements.txt",
        "model_only": True,
        "tokenizers_included": False,
        "packages": packages,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    requirements = parse_requirements(args.requirements)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--dest",
                str(args.output_dir),
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                "-r",
                str(args.requirements),
            ],
            check=True,
        )
    packages = verify_wheels(requirements, args.output_dir)
    manifest_path = write_manifest(packages, args.output_dir)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
