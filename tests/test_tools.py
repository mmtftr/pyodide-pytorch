from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import config  # noqa: E402
import fetch_lapack  # noqa: E402
import postprocess_wheel  # noqa: E402
import run_upstream_tests  # noqa: E402
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


def wasm_with_dynamic_libraries(
    *libraries: str, runtime_paths: tuple[str, ...] = ()
) -> bytes:
    def string(value: str) -> bytes:
        encoded = value.encode()
        return uleb(len(encoded)) + encoded

    needed = uleb(len(libraries)) + b"".join(string(name) for name in libraries)
    memory = uleb(0) + uleb(0) + uleb(0) + uleb(0)
    runtime = uleb(len(runtime_paths)) + b"".join(
        string(name) for name in runtime_paths
    )
    dylink = (
        uleb(1)
        + uleb(len(memory))
        + memory
        + uleb(2)
        + uleb(len(needed))
        + needed
        + uleb(5)
        + uleb(len(runtime))
        + runtime
    )
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

    def test_runtime_and_build_tool_versions_are_independent(self) -> None:
        manifest = config.load()
        manifest["pyodide"]["build_version"] = "9.8.7"
        self.assertEqual(config.validate(manifest), [])

    def test_upstream_manifest_is_pinned_and_auditable(self) -> None:
        manifest = json.loads(
            (ROOT / "tests" / "upstream_cpu_wasm.json").read_text(
                encoding="utf-8"
            )
        )
        values = config.flat_env(config.load())
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["pytorch_ref"], values["PYTORCH_REF"])
        self.assertTrue(
            all("==" in requirement for requirement in manifest["pypi_packages"])
        )

        expected = manifest["expected"]
        self.assertGreaterEqual(expected["total"], 600)
        self.assertEqual(expected["passed"], expected["total"])
        self.assertEqual(expected["skipped"], 0)
        self.assertEqual(expected["expected_failures"], 0)
        self.assertEqual(
            expected["collection_stubs"], len(manifest["collection_stubs"])
        )

        module_paths = [module["path"] for module in manifest["modules"]]
        self.assertEqual(len(module_paths), len(set(module_paths)))
        exclusions = []
        for module in manifest["modules"]:
            tests = module["tests"]
            self.assertTrue(tests)
            self.assertEqual(len(tests), len(set(tests)))
            if module.get("excluded_tests"):
                self.assertEqual(tests, ["*"])
            for exclusion in module.get("excluded_tests", []):
                self.assertTrue(exclusion["reason"].strip())
                exclusions.append((module["path"], exclusion["id"]))
        self.assertEqual(len(exclusions), len(set(exclusions)))
        self.assertEqual(expected["excluded"], len(exclusions))
        self.assertTrue(
            all(stub["reason"].strip() for stub in manifest["collection_stubs"])
        )
        policy = (ROOT / "docs" / "upstream-tests.md").read_text(encoding="utf-8")
        exclusion_inventory = policy.split(
            "## Explicitly excluded generated tests", maxsplit=1
        )[1].split("## Collection-only accommodation", maxsplit=1)[0]
        documented_exclusions = {}
        for line in exclusion_inventory.splitlines():
            if not line.startswith("| `"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            self.assertEqual(len(cells), 2)
            test_id = cells[0].strip("`")
            self.assertTrue(cells[1])
            documented_exclusions[test_id] = cells[1]
        self.assertEqual(
            set(documented_exclusions),
            {test_id for _, test_id in exclusions},
        )
        for stub in manifest["collection_stubs"]:
            self.assertIn(f"`{stub['module']}`", policy)

    def test_upstream_runner_rejects_unknown_exact_test_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test_sample.py").write_text(
                "import unittest\n"
                "class Sample(unittest.TestCase):\n"
                "    def test_present(self):\n"
                "        pass\n",
                encoding="utf-8",
            )
            manifest = {
                "modules": [
                    {
                        "path": "test_sample.py",
                        "tests": ["Sample.test_presnt"],
                    }
                ]
            }
            with self.assertRaisesRegex(
                ValueError, "closest generated IDs: Sample.test_present"
            ):
                run_upstream_tests.load_manifest_suite(manifest, root)

    def test_platform_tag_must_use_the_pyemscripten_abi(self) -> None:
        manifest = config.load()
        manifest["pyodide"]["platform_tag"] = "emscripten_5_0_3_wasm32"
        self.assertIn(
            "pyodide.platform_tag must be a pyemscripten wasm32 tag",
            config.validate(manifest),
        )

    def test_release_tag_uses_pinned_pyodide_version(self) -> None:
        manifest = config.load()
        manifest["release"]["tag"] = (
            "torch-prefix-pyodide-314.0.2-decoy-pyodide-9.9.9-r1"
        )
        self.assertIn(
            "release.tag must contain the pinned Pyodide version",
            config.validate(manifest),
        )

    def test_release_tag_rejects_invalid_git_ref_components(self) -> None:
        manifest = config.load()
        manifest["release"]["tag"] = "torch-a..b-pyodide-314.0.2-r1"
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

    def test_runtime_path_detection(self) -> None:
        self.assertEqual(
            validate_wheel.wasm_runtime_paths(
                wasm_with_dynamic_libraries(
                    "libopenblas.so",
                    runtime_paths=("$ORIGIN/../torch.libs",),
                )
            ),
            ["$ORIGIN/../torch.libs"],
        )

    def test_lapack_archive_rejects_unsafe_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "lapack.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../libopenblas.so", b"\0asm\x01\0\0\0")
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "unsafe LAPACK archive"):
                    fetch_lapack.safe_library_member(archive, "libopenblas.so")

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
            f"{values['PYODIDE_PLATFORM_TAG']}"
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
                wheel.writestr("torch/__init__.py", "")
                wheel.writestr("torch/version.py", "__version__ = '0'\n")
                wheel.writestr(
                    "torch/_C.test.so",
                    wasm_with_dynamic_libraries(
                        # Static target edges may preserve the same needed
                        # side module more than once.
                        "libopenblas.so",
                        "libopenblas.so",
                        runtime_paths=("$ORIGIN/../torch.libs",),
                    ),
                )
                wheel.writestr(
                    "torch.libs/libopenblas.so",
                    wasm_with_dynamic_libraries(runtime_paths=("$ORIGIN",)),
                )
                wheel.writestr("torch/lib/libtorch.a", b"archive")
                wheel.writestr("functorch/functorch.so", b"archive")
                wheel.writestr("torch-0.dist-info/METADATA", metadata)
                wheel.writestr("torch-0.dist-info/WHEEL", wheel_metadata)
                wheel.writestr("torch-0.dist-info/RECORD", "")
            destination = postprocess_wheel.repack(source, root / "out", 1_700_000_000)
            result = validate_wheel.validate(destination)
            self.assertEqual(
                result["dynamic_libraries"],
                ["libopenblas.so", "libopenblas.so"],
            )
            self.assertEqual(result["threading"], "single")
            with zipfile.ZipFile(destination) as wheel:
                names = wheel.namelist()
                self.assertNotIn("torch/lib/libtorch.a", names)
                self.assertNotIn("functorch/functorch.so", names)


if __name__ == "__main__":
    unittest.main()
