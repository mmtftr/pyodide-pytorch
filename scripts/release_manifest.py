#!/usr/bin/env python3
"""Write a machine-readable manifest next to a release wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import config


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    patches = sorted((ROOT / "patches" / "pytorch").glob("*.patch"))
    data = {
        "schema_version": 1,
        "builder_repository_commit": os.environ.get("GITHUB_SHA"),
        "configuration": config.load(),
        "wheel": {
            "filename": args.wheel.name,
            "sha256": sha256(args.wheel),
            "size": args.wheel.stat().st_size,
        },
        "inputs": {
            "build_script_sha256": sha256(ROOT / "scripts" / "build_wheel.sh"),
            "patches": [
                {"filename": patch.name, "sha256": sha256(patch)}
                for patch in patches
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
