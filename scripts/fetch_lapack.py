#!/usr/bin/env python3
"""Fetch and verify the LAPACK shared library matching the pinned Pyodide."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import config as manifest


CDN_TEMPLATE = "https://cdn.jsdelivr.net/pyodide/v{version}/full/{archive}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(path: Path, expected_sha256: str) -> None:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"LAPACK archive SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def safe_library_member(archive: zipfile.ZipFile, library: str) -> str:
    candidates: list[str] = []
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe LAPACK archive member: {name}")
        if not name.endswith("/") and path.name == library:
            candidates.append(name)
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one {library} in LAPACK archive, found {candidates}"
        )
    return candidates[0]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, temporary)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)


def prepare(cache_dir: Path, output_dir: Path, source: Path | None = None) -> Path:
    config = manifest.load()
    errors = manifest.validate(config)
    if errors:
        raise ValueError("; ".join(errors))
    values = manifest.flat_env(config)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_archive = cache_dir / values["LAPACK_ARCHIVE"]
    if source is not None:
        source = source.resolve()
        validate_archive(source, values["LAPACK_SHA256"])
        if source != cached_archive.resolve():
            shutil.copyfile(source, cached_archive)
    elif cached_archive.exists():
        try:
            validate_archive(cached_archive, values["LAPACK_SHA256"])
        except ValueError:
            cached_archive.unlink()
    if not cached_archive.exists():
        url = CDN_TEMPLATE.format(
            version=values["PYODIDE_VERSION"], archive=values["LAPACK_ARCHIVE"]
        )
        print(f"Downloading {url}")
        download(url, cached_archive)
    validate_archive(cached_archive, values["LAPACK_SHA256"])

    output_lib_dir = output_dir / "lib"
    output_lib_dir.mkdir(parents=True, exist_ok=True)
    destination = output_lib_dir / values["LAPACK_LIBRARY"]
    with zipfile.ZipFile(cached_archive) as archive:
        member = safe_library_member(archive, values["LAPACK_LIBRARY"])
        data = archive.read(member)
    if not data.startswith(b"\0asm\x01\0\0\0"):
        raise ValueError(f"{values['LAPACK_LIBRARY']} is not a WebAssembly module")
    destination.write_bytes(data)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="use a local archive instead of downloading (intended for tests)",
    )
    args = parser.parse_args()
    print(prepare(args.cache_dir, args.output_dir, args.source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
