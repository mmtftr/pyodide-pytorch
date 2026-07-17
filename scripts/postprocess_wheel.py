#!/usr/bin/env python3
"""Prune build-only payloads and repack a deterministic runtime wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import time
import zipfile
from pathlib import Path

import config as manifest


PRUNED_PREFIXES = (
    "caffe2/",
    "functorch/",
    "torch/bin/",
    "torch/include/",
    "torch/lib/",
    "torch/share/",
    "torch/test/",
)


def should_keep(name: str, record_path: str) -> bool:
    if name.endswith("/") or name == record_path:
        return False
    if name.startswith(PRUNED_PREFIXES):
        return False
    if name.endswith((".a", ".o")):
        return False
    return True


def add_provenance(metadata: bytes, values: dict[str, str]) -> bytes:
    text = metadata.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    separator = newline + newline
    headers, found, body = text.partition(separator)
    if not found:
        headers = text.rstrip("\r\n")
        body = ""
    headers = newline.join(
        line for line in headers.splitlines() if not line.startswith("X-Pyodide-")
    )
    additions = [
        f"X-Pyodide-PyTorch-Commit: {values['PYTORCH_REF']}",
        f"X-Pyodide-Runtime-Version: {values['PYODIDE_VERSION']}",
        f"X-Pyodide-Emscripten-Version: {values['EMSCRIPTEN_VERSION']}",
        "X-Pyodide-Threading: single",
    ]
    return (headers + newline + newline.join(additions) + separator + body).encode(
        "utf-8"
    )


def record_digest(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode("ascii")


def deterministic_info(original: zipfile.ZipInfo | None, name: str, epoch: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, time.gmtime(max(epoch, 315532800))[:6])
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.flag_bits = 0x800
    info.external_attr = (
        original.external_attr if original is not None else (0o100644 << 16)
    )
    return info


def repack(source: Path, output_dir: Path, epoch: int) -> Path:
    config = manifest.load()
    errors = manifest.validate(config)
    if errors:
        raise ValueError("; ".join(errors))
    values = manifest.flat_env(config)

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / source.name
    temporary_destination = destination.with_name(destination.name + ".tmp")
    if source.resolve() == destination.resolve():
        raise ValueError("input and output wheel paths must differ")

    records: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(source) as input_wheel:
        names = input_wheel.namelist()
        dist_infos = sorted(
            {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
        )
        if len(dist_infos) != 1:
            raise ValueError(f"expected one dist-info directory, found {dist_infos}")
        dist_info = dist_infos[0]
        metadata_path = f"{dist_info}/METADATA"
        record_path = f"{dist_info}/RECORD"
        if metadata_path not in names or record_path not in names:
            raise ValueError("wheel is missing METADATA or RECORD")

        with zipfile.ZipFile(
            temporary_destination,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as output_wheel:
            for name in sorted(names):
                if not should_keep(name, record_path):
                    continue
                data = input_wheel.read(name)
                if name == metadata_path:
                    data = add_provenance(data, values)
                if name.endswith(".so") and not data.startswith(b"\0asm"):
                    raise ValueError(f"native extension is not WebAssembly: {name}")
                info = deterministic_info(input_wheel.getinfo(name), name, epoch)
                output_wheel.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
                records.append((name, record_digest(data), str(len(data))))

            record_buffer = io.StringIO(newline="")
            writer = csv.writer(record_buffer, lineterminator="\n")
            writer.writerows(records)
            writer.writerow((record_path, "", ""))
            record_data = record_buffer.getvalue().encode("utf-8")
            output_wheel.writestr(
                deterministic_info(None, record_path, epoch),
                record_data,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    temporary_destination.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", "315532800")),
    )
    args = parser.parse_args()
    destination = repack(args.wheel, args.output_dir, args.source_date_epoch)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
