from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
import postprocess_wheel  # noqa: E402
import repair_pyodide_build  # noqa: E402
import validate_wheel  # noqa: E402


def uleb(value: int) -> bytes:
    result = bytearray()
    while True:
        part = value & 0x7F
        value >>= 7
        result.append(part | (0x80 if value else 0))
        if not value:
            return bytes(result)


def wasm_with_imported_memory(*, shared: bool) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return uleb(len(encoded)) + encoded

    flags = 0x3 if shared else 0x1
    payload = (
        uleb(1)
        + string("env")
        + string("memory")
        + b"\x02"
        + uleb(flags)
        + uleb(1)
        + uleb(2)
    )
    return b"\0asm\x01\0\0\0" + b"\x02" + uleb(len(payload)) + payload


def wasm_with_dynamic_libraries(*libraries: str) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return uleb(len(encoded)) + encoded

    needed = uleb(len(libraries)) + b"".join(string(name) for name in libraries)
    dylink = uleb(2) + uleb(len(needed)) + needed
    payload = string("dylink.0") + dylink
    return b"\0asm\x01\0\0\0" + b"\x00" + uleb(len(payload)) + payload


def wasm_with_function_imports(
    *names: str, exported: tuple[str, ...] = ()
) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return uleb(len(encoded)) + encoded

    entries = b"".join(
        string("env") + string(name) + b"\x00" + uleb(0) for name in names
    )
    import_payload = uleb(len(names)) + entries
    module = (
        b"\0asm\x01\0\0\0"
        + b"\x02"
        + uleb(len(import_payload))
        + import_payload
    )
    if exported:
        export_payload = uleb(len(exported)) + b"".join(
            string(name) + b"\x00" + uleb(0) for name in exported
        )
        module += b"\x07" + uleb(len(export_payload)) + export_payload
    return module


class ToolTests(unittest.TestCase):
    def test_manifest_is_valid(self) -> None:
        self.assertEqual(config.validate(config.load()), [])

    def test_shared_memory_detection(self) -> None:
        self.assertFalse(
            validate_wheel.wasm_uses_shared_memory(
                wasm_with_imported_memory(shared=False)
            )
        )
        self.assertTrue(
            validate_wheel.wasm_uses_shared_memory(
                wasm_with_imported_memory(shared=True)
            )
        )

    def test_dynamic_library_detection(self) -> None:
        self.assertEqual(
            validate_wheel.wasm_dynamic_libraries(
                wasm_with_dynamic_libraries("libtorch_python.so", "libshm.so")
            ),
            ["libtorch_python.so", "libshm.so"],
        )
        self.assertEqual(
            validate_wheel.wasm_dynamic_libraries(
                wasm_with_imported_memory(shared=False)
            ),
            [],
        )

    def test_unresolved_project_symbol_detection(self) -> None:
        data = wasm_with_function_imports(
            "invoke_vii",
            "cpuinfo_emscripten_init",
            "_ZN10onnx_torch9TypeProto11clear_valueEv",
            "_ZN3c1016already_resolvedEv",
            exported=("_ZN3c1016already_resolvedEv",),
        )
        self.assertEqual(
            validate_wheel.wasm_unresolved_project_symbols(data),
            [
                "_ZN10onnx_torch9TypeProto11clear_valueEv",
                "cpuinfo_emscripten_init",
            ],
        )

    def test_pyodide_toolchain_repair_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "Emscripten.cmake"
            self.assertTrue(repair_pyodide_build.install_toolchain(destination))
            self.assertFalse(repair_pyodide_build.install_toolchain(destination))
            destination.write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                repair_pyodide_build.install_toolchain(destination)

    def test_postprocess_prunes_and_produces_a_valid_wheel(self) -> None:
        values = config.flat_env(config.load())
        tag = (
            f"{values['PYTHON_TAG']}-{values['PYTHON_TAG']}-"
            f"emscripten_{values['EMSCRIPTEN_VERSION'].replace('.', '_')}_wasm32"
        )
        name = f"torch-test-{tag}.whl"
        metadata = """Metadata-Version: 2.1
Name: torch
Version: 0
Requires-Dist: filelock
Requires-Dist: fsspec
Requires-Dist: jinja2
Requires-Dist: networkx
Requires-Dist: sympy
Requires-Dist: typing-extensions

test
"""
        wheel_metadata = f"""Wheel-Version: 1.0
Generator: test
Root-Is-Purelib: false
Tag: {tag}

"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / name
            with zipfile.ZipFile(source, "w") as wheel:
                wheel.writestr(
                    "torch/_C.test.so", wasm_with_imported_memory(shared=False)
                )
                wheel.writestr("torch/lib/libtorch.a", b"archive")
                wheel.writestr("functorch/functorch.so", b"archive")
                wheel.writestr("torch-0.dist-info/METADATA", metadata)
                wheel.writestr("torch-0.dist-info/WHEEL", wheel_metadata)
                wheel.writestr("torch-0.dist-info/RECORD", "")
            destination = postprocess_wheel.repack(source, root / "out", 1_700_000_000)
            result = validate_wheel.validate(destination)
            self.assertEqual(result["threading"], "single")
            with zipfile.ZipFile(destination) as wheel:
                names = wheel.namelist()
                self.assertNotIn("torch/lib/libtorch.a", names)
                self.assertNotIn("functorch/functorch.so", names)


if __name__ == "__main__":
    unittest.main()
