#!/usr/bin/env python3
"""Validate ABI, provenance, RECORD integrity, and threadless Wasm memory."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path

import config as manifest


FORBIDDEN_PREFIXES = (
    "caffe2/",
    "functorch/",
    "torch/bin/",
    "torch/include/",
    "torch/lib/",
    "torch/share/",
    "torch/test/",
)
REQUIRED_DEPENDENCIES = {
    "filelock",
    "fsspec",
    "jinja2",
    "networkx",
    "sympy",
    "typing-extensions",
}
PROJECT_IMPORT_PREFIXES = (
    "_ZN2at",
    "_ZN3c10",
    "_ZN5torch",
    "_ZN6caffe2",
    "_ZN10onnx_torch",
    "_ZN21THManagedMapAllocator",
    "cpuinfo_",
)


class WasmFormatError(ValueError):
    pass


@dataclass
class Reader:
    data: bytes
    position: int = 0

    def byte(self) -> int:
        if self.position >= len(self.data):
            raise WasmFormatError("unexpected end of WebAssembly data")
        value = self.data[self.position]
        self.position += 1
        return value

    def uleb(self) -> int:
        result = 0
        shift = 0
        while True:
            value = self.byte()
            result |= (value & 0x7F) << shift
            if not value & 0x80:
                return result
            shift += 7
            if shift > 70:
                raise WasmFormatError("invalid unsigned LEB128 value")

    def take(self, count: int) -> bytes:
        end = self.position + count
        if end > len(self.data):
            raise WasmFormatError("section extends beyond WebAssembly module")
        value = self.data[self.position:end]
        self.position = end
        return value

    def string(self) -> str:
        return self.take(self.uleb()).decode("utf-8")


def read_limits(reader: Reader) -> bool:
    flags = reader.uleb()
    reader.uleb()
    if flags & 0x1:
        reader.uleb()
    return bool(flags & 0x2)


def wasm_uses_shared_memory(data: bytes) -> bool:
    if not data.startswith(b"\0asm\x01\0\0\0"):
        raise WasmFormatError("invalid WebAssembly magic or version")
    module = Reader(data, 8)
    while module.position < len(module.data):
        section_id = module.byte()
        payload = Reader(module.take(module.uleb()))
        if section_id == 0:
            name = payload.string()
            if name == "target_features" and b"atomics" in payload.data[payload.position :]:
                return True
        elif section_id == 2:
            for _ in range(payload.uleb()):
                payload.string()
                payload.string()
                kind = payload.byte()
                if kind == 0:
                    payload.uleb()
                elif kind == 1:
                    payload.byte()
                    read_limits(payload)
                elif kind == 2:
                    if read_limits(payload):
                        return True
                elif kind == 3:
                    payload.byte()
                    payload.byte()
                elif kind == 4:
                    payload.byte()
                    payload.uleb()
                else:
                    raise WasmFormatError(f"unknown import kind {kind}")
        elif section_id == 5:
            for _ in range(payload.uleb()):
                if read_limits(payload):
                    return True
    return False


def wasm_dynamic_libraries(data: bytes) -> list[str]:
    """Return DT_NEEDED-style entries from Emscripten's dylink.0 section."""
    if not data.startswith(b"\0asm\x01\0\0\0"):
        raise WasmFormatError("invalid WebAssembly magic or version")
    module = Reader(data, 8)
    libraries: list[str] = []
    while module.position < len(module.data):
        section_id = module.byte()
        payload = Reader(module.take(module.uleb()))
        if section_id != 0:
            continue
        name = payload.string()
        if name == "dylink":
            raise WasmFormatError("legacy dylink custom sections are unsupported")
        if name != "dylink.0":
            continue
        while payload.position < len(payload.data):
            subsection_type = payload.uleb()
            subsection = Reader(payload.take(payload.uleb()))
            if subsection_type != 2:
                continue
            libraries.extend(subsection.string() for _ in range(subsection.uleb()))
            if subsection.position != len(subsection.data):
                raise WasmFormatError("trailing data in dylink.0 needed subsection")
    return libraries


def wasm_imports(data: bytes) -> list[tuple[str, str, int]]:
    """Return (module, name, kind) entries from the Wasm import section."""
    if not data.startswith(b"\0asm\x01\0\0\0"):
        raise WasmFormatError("invalid WebAssembly magic or version")
    module = Reader(data, 8)
    imports: list[tuple[str, str, int]] = []
    while module.position < len(module.data):
        section_id = module.byte()
        payload = Reader(module.take(module.uleb()))
        if section_id != 2:
            continue
        for _ in range(payload.uleb()):
            imported_module = payload.string()
            name = payload.string()
            kind = payload.byte()
            imports.append((imported_module, name, kind))
            if kind == 0:
                payload.uleb()
            elif kind == 1:
                payload.byte()
                read_limits(payload)
            elif kind == 2:
                read_limits(payload)
            elif kind == 3:
                payload.byte()
                payload.byte()
            elif kind == 4:
                payload.byte()
                payload.uleb()
            else:
                raise WasmFormatError(f"unknown import kind {kind}")
    return imports


def wasm_exports(data: bytes) -> dict[str, int]:
    """Return exported symbol names and their external kinds."""
    if not data.startswith(b"\0asm\x01\0\0\0"):
        raise WasmFormatError("invalid WebAssembly magic or version")
    module = Reader(data, 8)
    exports: dict[str, int] = {}
    while module.position < len(module.data):
        section_id = module.byte()
        payload = Reader(module.take(module.uleb()))
        if section_id != 7:
            continue
        for _ in range(payload.uleb()):
            name = payload.string()
            kind = payload.byte()
            payload.uleb()
            exports[name] = kind
    return exports


def wasm_unresolved_project_symbols(data: bytes) -> list[str]:
    """Find known project-namespace imports that torch._C does not export."""
    exports = wasm_exports(data)
    return sorted(
        {
            name
            for _, name, _ in wasm_imports(data)
            if name.startswith(PROJECT_IMPORT_PREFIXES)
            and name not in exports
        }
    )


def urlsafe_digest(data: bytes) -> str:
    value = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + value.decode("ascii")


def normalized_requirement(requirement: str) -> str:
    name = re.split(r"[\s;(<>!=~\[]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_record(wheel: zipfile.ZipFile, record_path: str) -> None:
    rows = list(csv.reader(io.TextIOWrapper(wheel.open(record_path), encoding="utf-8")))
    indexed = {row[0]: row[1:] for row in rows}
    names = {name for name in wheel.namelist() if not name.endswith("/")}
    if set(indexed) != names:
        missing = sorted(names - set(indexed))
        stale = sorted(set(indexed) - names)
        raise ValueError(f"RECORD mismatch; missing={missing}, stale={stale}")
    for name in sorted(names - {record_path}):
        digest, size = indexed[name]
        data = wheel.read(name)
        if digest != urlsafe_digest(data) or size != str(len(data)):
            raise ValueError(f"RECORD digest or size mismatch for {name}")


def validate(path: Path) -> dict[str, object]:
    config = manifest.load()
    errors = manifest.validate(config)
    if errors:
        raise ValueError("; ".join(errors))
    values = manifest.flat_env(config)
    maximum_bytes = int(values["MAXIMUM_WHEEL_MIB"]) * 1024 * 1024
    if path.stat().st_size > maximum_bytes:
        raise ValueError(
            f"wheel is {path.stat().st_size / 1024 / 1024:.1f} MiB; "
            f"limit is {values['MAXIMUM_WHEEL_MIB']} MiB"
        )

    expected_tag = (
        f"{values['PYTHON_TAG']}-{values['PYTHON_TAG']}-"
        f"emscripten_{values['EMSCRIPTEN_VERSION'].replace('.', '_')}_wasm32"
    )
    if not path.name.endswith(f"-{expected_tag}.whl"):
        raise ValueError(f"wheel filename does not end with -{expected_tag}.whl")

    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate members")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe wheel member: {name}")
            if name.startswith(FORBIDDEN_PREFIXES) or name.endswith((".a", ".o")):
                raise ValueError(f"build-only payload remains in wheel: {name}")

        dist_infos = sorted(
            {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
        )
        if len(dist_infos) != 1:
            raise ValueError(f"expected one dist-info directory, found {dist_infos}")
        dist_info = dist_infos[0]
        metadata_path = f"{dist_info}/METADATA"
        wheel_path = f"{dist_info}/WHEEL"
        record_path = f"{dist_info}/RECORD"
        metadata = BytesParser(policy=compat32).parsebytes(wheel.read(metadata_path))
        wheel_metadata = BytesParser(policy=compat32).parsebytes(wheel.read(wheel_path))

        if expected_tag not in wheel_metadata.get_all("Tag", []):
            raise ValueError(f"WHEEL metadata is missing tag {expected_tag}")
        if wheel_metadata.get("Root-Is-Purelib", "").lower() != "false":
            raise ValueError("Root-Is-Purelib must be false")

        expected_headers = {
            "X-Pyodide-PyTorch-Commit": values["PYTORCH_REF"],
            "X-Pyodide-Runtime-Version": values["PYODIDE_VERSION"],
            "X-Pyodide-Emscripten-Version": values["EMSCRIPTEN_VERSION"],
            "X-Pyodide-Threading": "single",
        }
        for header, expected in expected_headers.items():
            if metadata.get(header) != expected:
                raise ValueError(f"{header} must be {expected!r}")

        dependencies = {
            normalized_requirement(item)
            for item in metadata.get_all("Requires-Dist", [])
        }
        missing_dependencies = REQUIRED_DEPENDENCIES - dependencies
        if missing_dependencies:
            raise ValueError(
                f"METADATA is missing dependencies: {sorted(missing_dependencies)}"
            )

        extensions = sorted(name for name in names if name.endswith(".so"))
        if len(extensions) != 1 or not extensions[0].startswith("torch/_C."):
            raise ValueError(f"expected only the torch._C extension, found {extensions}")
        extension_data = wheel.read(extensions[0])
        if wasm_uses_shared_memory(extension_data):
            raise ValueError("torch._C uses shared memory or the WebAssembly atomics feature")
        dynamic_libraries = wasm_dynamic_libraries(extension_data)
        if dynamic_libraries:
            raise ValueError(
                "torch._C is not self-contained; dynamic libraries: "
                + ", ".join(dynamic_libraries)
            )
        unresolved_project_symbols = wasm_unresolved_project_symbols(extension_data)
        if unresolved_project_symbols:
            raise ValueError(
                "torch._C has unresolved project symbols: "
                + ", ".join(unresolved_project_symbols)
            )

        validate_record(wheel, record_path)

    return {
        "wheel": path.name,
        "bytes": path.stat().st_size,
        "tag": expected_tag,
        "extensions": extensions,
        "dynamic_libraries": dynamic_libraries,
        "unresolved_project_symbols": unresolved_project_symbols,
        "threading": "single",
        "shared_memory": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.wheel), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
