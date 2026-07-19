from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
import postprocess_wheel  # noqa: E402
import repair_pyodide_build  # noqa: E402
import verify_release_artifact  # noqa: E402
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

    def test_release_tag_uses_pinned_pyodide_version(self) -> None:
        manifest = config.load()
        manifest["release"]["tag"] = (
            "torch-prefix-pyodide-0.24.1-decoy-pyodide-9.9.9-r1"
        )
        self.assertIn(
            "release.tag must contain the pinned Pyodide version",
            config.validate(manifest),
        )

    def test_release_tag_rejects_invalid_git_ref_components(self) -> None:
        manifest = config.load()
        manifest["release"]["tag"] = "torch-a..b-pyodide-0.24.1-r1"
        self.assertIn(
            "release.tag must match torch-*-pyodide-X.Y.Z[-rN]",
            config.validate(manifest),
        )

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

    def test_pyodide_cmake_command_mode_repair_is_idempotent(self) -> None:
        source = (
            '    elif cmd == "cmake":\n'
            f"{repair_pyodide_build.CMAKE_COMMAND_MODE_ORIGINAL}\n"
            "            return line\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "pywasmcross.py"
            destination.write_text(source, encoding="utf-8")
            original_digest = repair_pyodide_build.digest(destination)
            fixed_source = source.replace(
                repair_pyodide_build.CMAKE_COMMAND_MODE_ORIGINAL,
                repair_pyodide_build.CMAKE_COMMAND_MODE_FIXED,
            )
            fixed_digest = hashlib.sha256(fixed_source.encode()).hexdigest()
            with mock.patch.multiple(
                repair_pyodide_build,
                PYWASMCROSS_EXPECTED_SHA256=original_digest,
                PYWASMCROSS_FIXED_SHA256=fixed_digest,
            ):
                self.assertTrue(
                    repair_pyodide_build.install_cmake_command_mode_fix(destination)
                )
                self.assertFalse(
                    repair_pyodide_build.install_cmake_command_mode_fix(destination)
                )
                repaired = destination.read_text(encoding="utf-8")
                self.assertIn(
                    repair_pyodide_build.CMAKE_COMMAND_MODE_FIXED, repaired
                )
                self.assertNotIn(
                    repair_pyodide_build.CMAKE_COMMAND_MODE_ORIGINAL, repaired
                )

                destination.write_text("unexpected", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "has SHA-256"):
                    repair_pyodide_build.install_cmake_command_mode_fix(destination)

    def test_release_artifact_verification_binds_inputs_and_commit(self) -> None:
        builder_commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "torch-test.whl"
            wheel.write_bytes(b"tested wheel")
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            (root / f"{wheel.name}.sha256").write_text(
                f"{digest}  {wheel.name}\n", encoding="utf-8"
            )
            manifest = {
                "schema_version": 1,
                "builder_repository_commit": builder_commit,
                "configuration": config.load(),
                "inputs": verify_release_artifact.expected_inputs(),
                "wheel": {
                    "filename": wheel.name,
                    "sha256": digest,
                    "size": wheel.stat().st_size,
                },
            }
            (root / "build-manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            verified_wheel, errors = verify_release_artifact.verify(
                root, builder_commit
            )
            self.assertEqual(verified_wheel, wheel)
            self.assertEqual(errors, [])

            _, errors = verify_release_artifact.verify(root, "b" * 40)
            self.assertIn(
                "build manifest commit does not match the source workflow run",
                errors,
            )

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
