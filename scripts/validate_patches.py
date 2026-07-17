#!/usr/bin/env python3
"""Check patch ordering and unified-diff structure before touching upstream."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "patches" / "pytorch"
HUNK = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)


def validate_patch(path: Path) -> list[str]:
    errors: list[str] = []
    data = path.read_bytes()
    if b"\r\n" in data:
        errors.append(f"{path.name}: use LF line endings")
    lines = data.decode("utf-8").splitlines()
    if not lines or not lines[0].startswith("From "):
        errors.append(f"{path.name}: missing format-patch header")
    if not any(line.startswith("Subject: [PATCH ") for line in lines):
        errors.append(f"{path.name}: missing numbered patch subject")
    if not any(line.startswith("diff --git a/") for line in lines):
        errors.append(f"{path.name}: contains no file diff")

    current: tuple[int, int, int] | None = None
    for number, line in enumerate(lines + ["diff --git sentinel"], 1):
        match = HUNK.match(line)
        boundary = line.startswith("diff --git ") or line == "-- "
        if (match or boundary) and current is not None:
            declared_old, declared_new, start_line = current
            if (old_count, new_count) != (declared_old, declared_new):
                errors.append(
                    f"{path.name}:{start_line}: hunk declares "
                    f"{declared_old}/{declared_new} lines but contains "
                    f"{old_count}/{new_count}"
                )
            current = None
        if match:
            declared_old = int(match.group(2) or "1")
            declared_new = int(match.group(4) or "1")
            current = (declared_old, declared_new, number)
            old_count = 0
            new_count = 0
        elif current is not None and line and line[0] in " +-":
            if line.startswith(("--- ", "+++ ")):
                continue
            old_count += line[0] != "+"
            new_count += line[0] != "-"
    return errors


def main() -> int:
    patches = sorted(PATCH_DIR.glob("*.patch"))
    expected_prefixes = [f"{number:04d}-" for number in range(1, len(patches) + 1)]
    errors: list[str] = []
    if not patches:
        errors.append("no patches found")
    for path, expected in zip(patches, expected_prefixes):
        if not path.name.startswith(expected):
            errors.append(f"expected patch {path.name} to start with {expected}")
        errors.extend(validate_patch(path))
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"valid: {len(patches)} ordered PyTorch patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
