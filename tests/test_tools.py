from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import assemble_playground  # noqa: E402
import config  # noqa: E402
import fetch_lapack  # noqa: E402
import postprocess_wheel  # noqa: E402
import prepare_transformers_browser_deps  # noqa: E402
import run_upstream_tests  # noqa: E402
import stage_webgpu_sources  # noqa: E402
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

    def test_webgpu_sources_are_exactly_pinned_and_vendored(self) -> None:
        manifest = config.load()
        values = config.flat_env(manifest)
        self.assertEqual(
            values["TORCH_WEBGPU_REF"],
            "a4369ff0f61f4e58cbffb048cee85047b33dacba",
        )
        self.assertEqual(values["EMDAWNWEBGPU_RELEASE"], "v20251002.162335")
        self.assertRegex(values["EMDAWNWEBGPU_ARCHIVE_SHA512"], r"^[0-9a-f]{128}$")
        self.assertEqual(
            (ROOT / "vendor" / "torch-webgpu" / "COMMIT")
            .read_text(encoding="utf-8")
            .strip(),
            values["TORCH_WEBGPU_REF"],
        )
        version = (ROOT / "vendor" / "emdawnwebgpu" / "VERSION.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn(values["EMDAWNWEBGPU_RELEASE"], version)
        self.assertIn(values["EMDAWNWEBGPU_DAWN_REF"], version)
        for path in (
            ROOT / "vendor" / "torch-webgpu" / "LICENSE",
            ROOT / "vendor" / "emdawnwebgpu" / "webgpu" / "src" / "LICENSE",
            ROOT / "vendor" / "emdawnwebgpu" / "webgpu_cpp" / "LICENSE",
        ):
            self.assertTrue(path.is_file(), path)

    def test_webgpu_source_staging_is_network_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pytorch = Path(temporary)
            (pytorch / "torch").mkdir()
            (pytorch / "torch" / "CMakeLists.txt").write_text(
                "# fixture\n", encoding="utf-8"
            )
            original = sys.argv
            try:
                sys.argv = ["stage_webgpu_sources.py", str(pytorch)]
                self.assertEqual(stage_webgpu_sources.main(), 0)
            finally:
                sys.argv = original
            self.assertTrue(
                (
                    pytorch
                    / "third_party"
                    / "torch-webgpu"
                    / "csrc"
                    / "ops"
                    / "binary.cpp"
                ).is_file()
            )
            self.assertTrue(
                (
                    pytorch
                    / "third_party"
                    / "emdawnwebgpu"
                    / "webgpu_cpp"
                    / "include"
                    / "webgpu"
                    / "webgpu_cpp.h"
                ).is_file()
            )
            project_kernels = (
                pytorch
                / "third_party"
                / "pyodide-pytorch-webgpu"
                / "llm_kernels"
            )
            self.assertTrue((project_kernels / "attention.cpp").is_file())
            self.assertTrue((project_kernels / "argmax.cpp").is_file())
            self.assertTrue((project_kernels / "boolean.cpp").is_file())
            self.assertTrue((project_kernels / "creation.cpp").is_file())
            self.assertTrue((project_kernels / "gemma_rms_norm.cpp").is_file())
            self.assertTrue((project_kernels / "generation_control.cpp").is_file())
            self.assertTrue((project_kernels / "generation_control.h").is_file())
            self.assertTrue((project_kernels / "generation_long.cpp").is_file())
            self.assertTrue((project_kernels / "masking.cpp").is_file())
            self.assertTrue((project_kernels / "normalization.cpp").is_file())
            self.assertTrue((project_kernels / "layer_norm.wgsl").is_file())
            self.assertTrue((project_kernels / "scalar_binary.cpp").is_file())
            self.assertTrue((project_kernels / "swiglu.cpp").is_file())
            self.assertTrue((project_kernels / "kv_cache.cpp").is_file())
            self.assertTrue((project_kernels / "q8_linear.cpp").is_file())
            embedded = (project_kernels / "embedded_shaders.h").read_text(
                encoding="utf-8"
            )
            self.assertIn("inline constexpr char kArange[]", embedded)
            self.assertIn("inline constexpr char kArgmax[]", embedded)
            self.assertIn("inline constexpr char kAllBool[]", embedded)
            self.assertIn("inline constexpr char kAnyBool[]", embedded)
            self.assertIn("inline constexpr char kBitwiseNotBool[]", embedded)
            self.assertIn("inline constexpr char kEqScalar[]", embedded)
            self.assertIn("inline constexpr char kFill[]", embedded)
            self.assertIn("inline constexpr char kGemmaRmsNorm[]", embedded)
            self.assertIn("inline constexpr char kGtTensor[]", embedded)
            self.assertIn("inline constexpr char kSdpa[]", embedded)
            self.assertIn("inline constexpr char kLayerNorm[]", embedded)
            self.assertIn("inline constexpr char kLinear[]", embedded)
            self.assertIn("inline constexpr char kLongCumsum[]", embedded)
            self.assertIn("inline constexpr char kLongIsin[]", embedded)
            self.assertIn("inline constexpr char kLongLtScalar[]", embedded)
            self.assertIn("inline constexpr char kMeanDim[]", embedded)
            self.assertIn("inline constexpr char kMulBoolInplace[]", embedded)
            self.assertIn("inline constexpr char kMulBoolTensor[]", embedded)
            self.assertIn("inline constexpr char kNeTensor[]", embedded)
            self.assertIn("inline constexpr char kIntToFloat[]", embedded)
            self.assertIn("inline constexpr char kKvCacheUpdate[]", embedded)
            self.assertIn("inline constexpr char kLinearGemvQ8S4[]", embedded)
            self.assertIn("inline constexpr char kScalarBinary[]", embedded)
            self.assertIn("inline constexpr char kTriangular[]", embedded)
            self.assertIn("inline constexpr char kWhereFloat[]", embedded)
            self.assertIn("inline constexpr char kSwiGluGemv[]", embedded)
            self.assertIn(
                "inline constexpr char kSwiGluGemvSubgroupS4[]", embedded
            )
            scalar_binary = (project_kernels / "scalar_binary.cpp").read_text(
                encoding="utf-8"
            )
            for schema in (
                "add.Scalar",
                "add_.Scalar",
                "add.Scalar_out",
                "sub.Scalar",
                "sub_.Scalar",
                "sub.Scalar_out",
                "mul.Scalar",
                "mul_.Scalar",
                "mul.Scalar_out",
                "div.Scalar",
                "div_.Scalar",
                "div.Scalar_out",
            ):
                self.assertIn(f'module.impl("{schema}"', scalar_binary)

            # A second staging pass must be a true no-op for unchanged shader
            # inputs; touching this header recompiles every kernel owner.
            embedded_path = project_kernels / "embedded_shaders.h"
            stable_mtime = 1_700_000_000_000_000_000
            os.utime(embedded_path, ns=(stable_mtime, stable_mtime))
            stale = project_kernels / "stale-input.cpp"
            stale.write_text("stale\n", encoding="utf-8")
            original = sys.argv
            try:
                sys.argv = ["stage_webgpu_sources.py", str(pytorch)]
                self.assertEqual(stage_webgpu_sources.main(), 0)
            finally:
                sys.argv = original
            self.assertEqual(embedded_path.stat().st_mtime_ns, stable_mtime)
            self.assertFalse(stale.exists())

    def test_native_layer_norm_kernel_contract_is_integrated(self) -> None:
        source = (
            ROOT / "webgpu" / "llm_kernels" / "normalization.cpp"
        ).read_text(encoding="utf-8")
        shader = (
            ROOT / "webgpu" / "llm_kernels" / "layer_norm.wgsl"
        ).read_text(encoding="utf-8")
        patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0010-add-experimental-browser-webgpu-backend.patch"
        ).read_text(encoding="utf-8")
        raw_browser_test = (
            ROOT / "tests" / "webgpu-layer-norm-wgsl.mjs"
        ).read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            stage_webgpu_sources.SHADERS["layer_norm.wgsl"], "kLayerNorm"
        )
        self.assertIn('module.impl("native_layer_norm"', source)
        self.assertIn("input.dim() <= 8", source)
        self.assertIn("partial_count", shader)
        self.assertIn("partial_mean", shader)
        self.assertIn("partial_m2", shader)
        self.assertNotIn("square_sum", shader)
        self.assertIn(
            "pyodide-pytorch-webgpu/llm_kernels/normalization.cpp",
            patch,
        )
        self.assertIn("stable-affine-tail", raw_browser_test)
        self.assertIn("no-affine-repeated-read-binding", raw_browser_test)
        self.assertIn("node tests/webgpu-layer-norm-wgsl.mjs", workflow)

    def test_decode_swiglu_kernel_contract_is_integrated(self) -> None:
        source = (
            ROOT / "webgpu" / "llm_kernels" / "swiglu.cpp"
        ).read_text(encoding="utf-8")
        portable = (
            ROOT / "webgpu" / "llm_kernels" / "swiglu_gemv.wgsl"
        ).read_text(encoding="utf-8")
        subgroup = (
            ROOT
            / "webgpu"
            / "llm_kernels"
            / "swiglu_gemv_subgroup_s4.wgsl"
        ).read_text(encoding="utf-8")
        raw_browser_test = (
            ROOT / "tests" / "webgpu-swiglu-wgsl.mjs"
        ).read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )
        patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0010-add-experimental-browser-webgpu-backend.patch"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            stage_webgpu_sources.SHADERS["swiglu_gemv.wgsl"],
            "kSwiGluGemv",
        )
        self.assertEqual(
            stage_webgpu_sources.SHADERS["swiglu_gemv_subgroup_s4.wgsl"],
            "kSwiGluGemvSubgroupS4",
        )
        self.assertIn("fused_swiglu(Tensor input", source)
        self.assertIn("rows == 1", source)
        self.assertIn("fusedSwiGluVariant", source)
        self.assertIn("silu(gate) * up", portable)
        self.assertIn("enable subgroups", subgroup)
        self.assertIn("subgroupAdd(gate)", subgroup)
        self.assertIn("llm_kernels/swiglu.cpp", patch)
        self.assertIn("decode-tail-no-bias-repeated-read-binding", raw_browser_test)
        self.assertIn("independent-biases-and-storage-offsets", raw_browser_test)
        self.assertIn("dispatches: 1", raw_browser_test)
        self.assertIn("node tests/webgpu-swiglu-wgsl.mjs", workflow)

    def test_preallocated_kv_cache_kernel_contract_is_integrated(self) -> None:
        source = (
            ROOT / "webgpu" / "llm_kernels" / "kv_cache.cpp"
        ).read_text(encoding="utf-8")
        shader = (
            ROOT / "webgpu" / "llm_kernels" / "kv_cache_update.wgsl"
        ).read_text(encoding="utf-8")
        bootstrap = (
            ROOT / "site" / "transformers_browser_bootstrap.py"
        ).read_text(encoding="utf-8")
        raw_browser_test = (
            ROOT / "tests" / "webgpu-kv-cache-wgsl.mjs"
        ).read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )
        patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0010-add-experimental-browser-webgpu-backend.patch"
        ).read_text(encoding="utf-8")

        self.assertEqual(
            stage_webgpu_sources.SHADERS["kv_cache_update.wgsl"],
            "kKvCacheUpdate",
        )
        self.assertIn("update_kv_cache_(Tensor(a!) key_cache", source)
        self.assertIn("params.position_words == 2u", shader)
        self.assertIn("key_cache[key_destination]", shader)
        self.assertIn("value_cache[value_destination]", shader)
        self.assertIn("llm_kernels/kv_cache.cpp", patch)
        self.assertIn(
            'cache_implementation="webgpu_preallocated"', bootstrap
        )
        self.assertIn("return self._publish_prefix", bootstrap)
        cache_class_source = bootstrap.split(
            "def _make_webgpu_preallocated_cache_class", 1
        )[1].split("def _preallocated_kv_diagnostics", 1)[0]
        self.assertNotIn("torch.cat", cache_class_source)
        self.assertIn("long-indexed-two-step-prefix", raw_browser_test)
        self.assertIn("dispatches: 1", raw_browser_test)
        self.assertIn("node tests/webgpu-kv-cache-wgsl.mjs", workflow)

    def test_webgpu_bool_mask_contract_is_integrated(self) -> None:
        source = (
            ROOT / "webgpu" / "llm_kernels" / "boolean.cpp"
        ).read_text(encoding="utf-8")
        eq_shader = (
            ROOT / "webgpu" / "llm_kernels" / "eq_scalar.wgsl"
        ).read_text(encoding="utf-8")
        all_shader = (
            ROOT / "webgpu" / "llm_kernels" / "all_bool.wgsl"
        ).read_text(encoding="utf-8")
        patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0010-add-experimental-browser-webgpu-backend.patch"
        ).read_text(encoding="utf-8")
        raw_browser_test = (
            ROOT / "tests" / "webgpu-bool-wgsl.mjs"
        ).read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )

        self.assertEqual(
            stage_webgpu_sources.SHADERS["eq_scalar.wgsl"], "kEqScalar"
        )
        self.assertEqual(
            stage_webgpu_sources.SHADERS["all_bool.wgsl"], "kAllBool"
        )
        self.assertIn('module.impl("eq.Scalar"', source)
        self.assertIn('module.impl("all"', source)
        self.assertIn("output_word * 4u", eq_shader)
        self.assertIn("input[index >> 2u]", all_shader)
        self.assertNotIn("array<bool>", eq_shader + all_shader)
        self.assertIn("llm_kernels/boolean.cpp", patch)
        self.assertIn("torch.Tensor.__bool__ = _tensor_bool", patch)
        self.assertIn("torch.bool: np.bool_", patch)
        self.assertIn("packed eq.Scalar mismatch", raw_browser_test)
        self.assertIn("packed Bool fill mismatch", raw_browser_test)
        self.assertIn("node tests/webgpu-bool-wgsl.mjs", workflow)

    def test_webgpu_build_keeps_side_module_em_js_exports(self) -> None:
        build_script = (ROOT / "scripts" / "build_wheel.sh").read_text(
            encoding="utf-8"
        )
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("--exports=whole_archive", build_script)
        patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0010-add-experimental-browser-webgpu-backend.patch"
        ).read_text(encoding="utf-8")
        self.assertIn('"webgpu/licenses/*.txt"', patch)
        self.assertIn("llm_kernels/scalar_binary.cpp", patch)
        self.assertIn("node tests/webgpu-scalar-binary-wgsl.mjs", workflow)
        self.assertIn("copy_strided(temporary, output)", patch)
        architecture = (
            ROOT / "docs" / "webgpu-browser-architecture.md"
        ).read_text(encoding="utf-8")
        self.assertIn("SIDE_MODULE=1", architecture)

    def test_build_environment_is_stable_and_fully_pinned(self) -> None:
        build_script = (ROOT / "scripts" / "build_wheel.sh").read_text(
            encoding="utf-8"
        )
        constraints = (ROOT / "config" / "build-constraints.txt").read_text(
            encoding="utf-8"
        )
        workflow = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )
        host_protoc_patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0003-use-a-host-protoc-when-cross-compiling.patch"
        ).read_text(encoding="utf-8")
        project_hook_patch = (
            ROOT
            / "patches"
            / "pytorch"
            / "0012-pass-project-hooks-before-cmake-project.patch"
        ).read_text(encoding="utf-8")

        # Pyodide's isolated build environment overlays wasm32-specific NumPy
        # headers. Bypassing it compiles torch against native x86-64 NumPy ABI
        # metadata and corrupts torch <-> NumPy conversion at runtime.
        self.assertNotIn("--no-isolation", build_script)
        self.assertIn("wasm32-specific files", build_script)
        self.assertIn("CMAKE_MAKE_PROGRAM:FILEPATH=$stable_ninja", build_script)
        self.assertIn(
            'grep -Fqx "$expected_ninja_cache" "$cmake_cache"',
            build_script,
        )
        self.assertIn('CCACHE_COMPILERCHECK:-content', build_script)
        self.assertIn('CCACHE_BASEDIR:-$source_dir', build_script)
        self.assertIn('export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-', build_script)
        self.assertIn("SOURCE_DATE_EPOCH must be a non-negative integer", build_script)
        self.assertIn("pywasmcross_env.json", build_script)
        self.assertIn("CCACHE_EXTRAFILES", build_script)
        self.assertIn('"::") export CCACHE_EXTRAFILES', build_script)
        self.assertIn("CMAKE_PROJECT_INCLUDE_BEFORE", build_script)
        self.assertIn("PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE", build_script)
        self.assertIn("'scripts/cmake/**'", workflow)
        self.assertIn(
            "cmake_hooks_tree_sha256",
            verify_release_artifact.expected_inputs(),
        )
        self.assertIn("protobuf_BUILD_PROTOC_BINARIES OFF", host_protoc_patch)
        self.assertIn('"CMAKE_PROJECT_INCLUDE_BEFORE"', project_hook_patch)
        self.assertIn('"CMAKE_PROJECT_INCLUDE"', project_hook_patch)
        self.assertIn(
            "build_options[project_hook] = project_hook_value",
            project_hook_patch,
        )
        self.assertIn("-r config/build-constraints.txt", workflow)
        self.assertNotIn("system_headers", workflow)
        for requirement in (
            "cmake",
            "ninja",
            "numpy",
            "packaging",
            "pyyaml",
            "requests",
            "setuptools",
            "six",
            "typing-extensions",
            "wheel",
        ):
            self.assertRegex(
                constraints,
                rf"(?m)^{re.escape(requirement)}==[^\s]+$",
            )

    def test_pyodide_side_module_flag_normalizer_is_idempotent(self) -> None:
        bootstrap = (
            ROOT
            / "scripts"
            / "cmake"
            / "pyodide_project_include_before.cmake"
        )
        normalizer = (
            ROOT
            / "scripts"
            / "cmake"
            / "normalize_pyodide_side_module_flags.cmake"
        )
        self.assertTrue(bootstrap.is_file())
        self.assertTrue(normalizer.is_file())

        side_module_cflags = "-O2 -g0 -fPIC -DFAKE_SIDE_C=1"
        side_module_cxxflags = (
            "-O2 -g0 -fPIC -fexceptions -DFAKE_SIDE_CXX=1 -Oz"
        )
        side_module_ldflags = (
            "-O2 -g0 -L/target/lib -Wl,--as-needed -Oz"
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            build = root / "build"
            source.mkdir()
            nested = source / "nested"
            nested.mkdir()
            toolchain = root / "repeating-toolchain.cmake"
            caller_before = root / "caller-before.cmake"
            caller_after = root / "caller-after.cmake"

            toolchain.write_text(
                "foreach(_repeat RANGE 1 4)\n"
                "  set(CMAKE_C_FLAGS \"${CMAKE_C_FLAGS} "
                "$ENV{SIDE_MODULE_CFLAGS}\")\n"
                "  set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} "
                "$ENV{SIDE_MODULE_CXXFLAGS}\")\n"
                "  set(CMAKE_SHARED_LINKER_FLAGS "
                "\"${CMAKE_SHARED_LINKER_FLAGS} $ENV{SIDE_MODULE_LDFLAGS}\")\n"
                "  set(CMAKE_MODULE_LINKER_FLAGS "
                "\"${CMAKE_MODULE_LINKER_FLAGS} $ENV{SIDE_MODULE_LDFLAGS}\")\n"
                "  set(CMAKE_SHARED_LINKER_FLAGS_INIT "
                "\"${CMAKE_SHARED_LINKER_FLAGS_INIT} "
                "$ENV{SIDE_MODULE_LDFLAGS}\")\n"
                "  set(CMAKE_MODULE_LINKER_FLAGS_INIT "
                "\"${CMAKE_MODULE_LINKER_FLAGS_INIT} "
                "$ENV{SIDE_MODULE_LDFLAGS}\")\n"
                "endforeach()\n",
                encoding="utf-8",
            )
            caller_before.write_text(
                'set(PYTORCH_TEST_BEFORE_HOOK "yes")\n',
                encoding="utf-8",
            )
            caller_after.write_text(
                'set(PYTORCH_TEST_AFTER_HOOK "yes")\n'
                "if(NOT DEFINED PYTORCH_TEST_SCOPED_FLAGS_ADDED)\n"
                '  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} '
                '$ENV{SIDE_MODULE_CXXFLAGS} -fafter")\n'
                '  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} '
                '$ENV{SIDE_MODULE_LDFLAGS} -Wl,--after")\n'
                '  set(PYTORCH_TEST_SCOPED_FLAGS_ADDED "yes")\n'
                "endif()\n",
                encoding="utf-8",
            )
            (source / "probe.c").write_text("int c_probe(void) { return 1; }\n")
            (source / "probe.cpp").write_text(
                "int cxx_probe() { return 2; }\n"
            )
            (nested / "nested.c").write_text(
                "int nested_c_probe(void) { return 3; }\n"
            )
            (nested / "nested.cpp").write_text(
                "int nested_cxx_probe() { return 4; }\n"
            )
            (nested / "CMakeLists.txt").write_text(
                "project(nested_side_module_normalizer LANGUAGES C CXX)\n"
                "add_library(nested_probe STATIC nested.c nested.cpp)\n",
                encoding="utf-8",
            )
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(side_module_normalizer LANGUAGES C CXX)\n"
                "add_library(probe STATIC probe.c probe.cpp)\n"
                "add_subdirectory(nested)\n"
                "file(WRITE \"${CMAKE_BINARY_DIR}/observed.txt\"\n"
                "  \"C=${CMAKE_C_FLAGS}\\n\"\n"
                "  \"CXX=${CMAKE_CXX_FLAGS}\\n\"\n"
                "  \"SHARED=${CMAKE_SHARED_LINKER_FLAGS}\\n\"\n"
                "  \"MODULE=${CMAKE_MODULE_LINKER_FLAGS}\\n\"\n"
                "  \"SHARED_INIT=${CMAKE_SHARED_LINKER_FLAGS_INIT}\\n\"\n"
                "  \"MODULE_INIT=${CMAKE_MODULE_LINKER_FLAGS_INIT}\\n\"\n"
                "  \"BEFORE=${PYTORCH_TEST_BEFORE_HOOK}\\n\"\n"
                "  \"AFTER=${PYTORCH_TEST_AFTER_HOOK}\\n\"\n"
                ")\n",
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment.update(
                {
                    "SIDE_MODULE_CFLAGS": side_module_cflags,
                    "SIDE_MODULE_CXXFLAGS": side_module_cxxflags,
                    "SIDE_MODULE_LDFLAGS": side_module_ldflags,
                    "PYTORCH_CHAINED_CMAKE_PROJECT_INCLUDE_BEFORE": str(
                        caller_before
                    ),
                }
            )
            command = [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                "Unix Makefiles",
                f"-DCMAKE_TOOLCHAIN_FILE:FILEPATH={toolchain}",
                f"-DCMAKE_PROJECT_INCLUDE_BEFORE:FILEPATH={bootstrap}",
                f"-DCMAKE_PROJECT_INCLUDE:FILEPATH={caller_after}",
                "-DCMAKE_C_FLAGS:STRING=-Wall",
                "-DCMAKE_CXX_FLAGS:STRING=-Wextra",
                "-DCMAKE_SHARED_LINKER_FLAGS:STRING=-Wl,--keep",
                "-DCMAKE_MODULE_LINKER_FLAGS:STRING=-Wl,--module",
            ]

            def configure() -> tuple[str, str]:
                result = subprocess.run(
                    command,
                    check=False,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                generated_flags = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in sorted(build.rglob("flags.make"))
                )
                return (
                    (build / "observed.txt").read_text(encoding="utf-8"),
                    generated_flags,
                )

            first, first_ninja = configure()
            second, second_ninja = configure()
            self.assertEqual(second, first)
            self.assertEqual(second_ninja, first_ninja)
            for injected_flags in (
                side_module_cflags,
                side_module_cxxflags,
                side_module_ldflags,
            ):
                self.assertNotIn(injected_flags, first)
                self.assertNotIn(injected_flags, first_ninja)

            observed = dict(
                line.split("=", 1) for line in first.splitlines()
            )
            self.assertEqual(shlex.split(observed["C"]), ["-Wall"])
            self.assertEqual(
                shlex.split(observed["CXX"]), ["-Wextra", "-fafter"]
            )
            self.assertEqual(
                shlex.split(observed["SHARED"]),
                ["-Wl,--keep", "-Wl,--after"],
            )
            self.assertEqual(
                shlex.split(observed["MODULE"]), ["-Wl,--module"]
            )
            self.assertEqual(observed["BEFORE"], "yes")
            self.assertEqual(observed["AFTER"], "yes")

            cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
            for injected_flags in (
                side_module_cflags,
                side_module_cxxflags,
                side_module_ldflags,
            ):
                self.assertNotIn(injected_flags, cache)
            self.assertIn("CMAKE_C_FLAGS:STRING=-Wall", cache)
            self.assertIn("CMAKE_CXX_FLAGS:STRING=-Wextra", cache)
            self.assertNotIn("CMAKE_CXX_FLAGS:STRING=-Wextra -fafter", cache)
            self.assertIn(
                "CMAKE_SHARED_LINKER_FLAGS:STRING=-Wl,--keep",
                cache,
            )
            self.assertNotIn(
                "CMAKE_SHARED_LINKER_FLAGS:STRING=-Wl,--keep -Wl,--after",
                cache,
            )
            self.assertIn(
                "CMAKE_MODULE_LINKER_FLAGS:STRING=-Wl,--module", cache
            )

    def test_playground_exposes_verified_transformer_benchmark(self) -> None:
        index = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        app = (ROOT / "site" / "app.js").read_text(encoding="utf-8")
        service_worker = (ROOT / "site" / "service-worker.js").read_text(
            encoding="utf-8"
        )
        transformers_bootstrap = (
            ROOT / "site" / "transformers_browser_bootstrap.py"
        ).read_text(encoding="utf-8")
        pages = (ROOT / ".github" / "workflows" / "pages.yml").read_text(
            encoding="utf-8"
        )
        assembler = (ROOT / "scripts" / "assemble_playground.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('value="transformerBenchmark"', index)
        self.assertIn("transformerBenchmark:", app)
        self.assertIn("F.scaled_dot_product_attention", app)
        self.assertIn("await torch.webgpu.synchronize()", app)
        self.assertIn("await torch.webgpu.to_cpu_async(gpu_logits)", app)
        worker = (ROOT / "site" / "worker.js").read_text(encoding="utf-8")
        self.assertIn('const ASSET_VERSION = "11"', app)
        self.assertIn('<script type="module" src="./app.js?v=11"></script>', index)
        self.assertIn("shell-v11", service_worker)
        self.assertIn("runtime-v7", service_worker)
        self.assertIn("./app.js?v=11", service_worker)
        self.assertIn("./worker.js?v=11", service_worker)
        self.assertIn("./transformers_browser_bootstrap.py", service_worker)
        self.assertIn("./transformers_gemma2_webgpu.py", service_worker)
        self.assertIn("./transformers_q8.py", service_worker)
        self.assertIn("event.respondWith(fetch(request))", service_worker)
        self.assertIn("isManifest || isTransformersManifest", service_worker)
        self.assertIn('crypto.subtle.digest("SHA-256"', worker)
        self.assertIn("wheelSha256 !== config.wheelSha256", worker)
        self.assertNotIn("micropip.install", worker)
        self.assertIn('["filelock", "huggingface-hub", "transformers"]', worker)
        self.assertIn('"./transformers_browser_bootstrap.py"', worker)
        self.assertIn('"./transformers_gemma2_webgpu.py"', worker)
        self.assertIn('"./transformers_q8.py"', worker)
        self.assertIn(
            "disable_optional_gguf_without_tokenizers()",
            worker,
        )
        self.assertIn("enable_webgpu_rms_norm_fusion()", worker)
        self.assertIn("webgpu_rms_norm_fusion", worker)
        self.assertIn(
            "enable_webgpu_rotary_scaling_compatibility()",
            worker,
        )
        self.assertIn("webgpu_rotary_scaling_compatibility", worker)
        self.assertIn("enable_webgpu_swiglu_fusion()", worker)
        self.assertIn("webgpu_swiglu_fusion", worker)
        self.assertIn("enable_webgpu_opt_sdpa_mask_compatibility()", worker)
        self.assertIn("webgpu_opt_sdpa_mask_compatibility", worker)
        self.assertIn("enable_webgpu_preallocated_kv_cache()", worker)
        self.assertIn("webgpu_preallocated_kv_cache", worker)
        self.assertIn("convert_linear_modules_q8_", worker)
        self.assertIn("webgpu_q8_linear", worker)
        self.assertIn("enable_webgpu_gemma2_rms_norm()", worker)
        self.assertIn("enable_webgpu_gemma2_scalar_normalizer()", worker)
        self.assertIn("webgpu_gemma2_rms_norm", worker)
        self.assertIn("webgpu_gemma2_scalar_normalizer", worker)
        self.assertIn('"operator_available": callable(torch.ops.webgpu.q8_linear)', worker)
        self.assertIn('"device_supported": webgpu_q8_device_supported', worker)
        self.assertIn(
            '"./runtime/transformers/transformers-browser-manifest.json"',
            worker,
        )
        self.assertIn("transformersPackages.get(\"transformers\").version", worker)
        self.assertIn('value="transformersTiny"', index)
        self.assertIn('filename: "transformers_tiny.py"', app)
        self.assertIn('print("model:", type(model).__name__)', app)
        self.assertIn('print("WebGPU forward: passed")', app)
        self.assertIn("enable_webgpu_rms_norm_fusion()", app)
        self.assertIn("enable_webgpu_rotary_scaling_compatibility()", app)
        self.assertIn("enable_webgpu_swiglu_fusion()", app)
        self.assertIn("enable_webgpu_opt_sdpa_mask_compatibility()", app)
        self.assertIn("enable_webgpu_preallocated_kv_cache()", app)
        self.assertIn(
            "GGUF loading is disabled in the model-only browser stack",
            transformers_bootstrap,
        )
        self.assertIn("supports exactly Transformers", transformers_bootstrap)
        for class_name in (
            "Qwen2RMSNorm",
            "LlamaRMSNorm",
            "MistralRMSNorm",
            "Phi3RMSNorm",
        ):
            self.assertIn(class_name, transformers_bootstrap)
        for class_name in ("Qwen2RotaryEmbedding", "LlamaRotaryEmbedding"):
            self.assertIn(class_name, transformers_bootstrap)
        for class_name in ("Qwen2MLP", "LlamaMLP", "MistralMLP"):
            self.assertIn(class_name, transformers_bootstrap)
        self.assertIn('"transformers_browser_bootstrap.py"', assembler)
        self.assertIn('"transformers_gemma2_webgpu.py"', assembler)
        self.assertIn('"transformers_q8.py"', assembler)
        self.assertIn("actions/download-artifact@v8", pages)
        self.assertNotIn("FALLBACK_RELEASE", app)
        self.assertIn("Resolve newest complete Publisher run", pages)
        self.assertIn("scripts/assemble_playground.py", pages)

    def test_transformers_browser_acceptance_is_hermetic_and_required(self) -> None:
        fixture = json.loads(
            (ROOT / "tests" / "fixtures" / "transformers_tiny.json").read_text(
                encoding="utf-8"
            )
        )
        page = (ROOT / "tests" / "transformers-webgpu.html").read_text(
            encoding="utf-8"
        )
        harness = (ROOT / "tests" / "transformers-webgpu.mjs").read_text(
            encoding="utf-8"
        )
        requirements = (
            ROOT / "config" / "transformers-browser-requirements.txt"
        ).read_text(encoding="utf-8")
        build = (ROOT / ".github" / "workflows" / "build.yml").read_text(
            encoding="utf-8"
        )
        publisher = (
            ROOT / ".github" / "workflows" / "publish-release.yml"
        ).read_text(encoding="utf-8")
        pages = (ROOT / ".github" / "workflows" / "pages.yml").read_text(
            encoding="utf-8"
        )
        assembler = (ROOT / "scripts" / "assemble_playground.py").read_text(
            encoding="utf-8"
        )

        self.assertEqual(fixture["transformers_version"], "4.46.3")
        self.assertEqual(
            [model["name"] for model in fixture["models"]],
            ["qwen2", "llama", "mistral", "gpt2", "bert", "phi3", "opt"],
        )
        self.assertIn("transformers==4.46.3", requirements)
        self.assertIn("huggingface-hub==0.26.2", requirements)
        self.assertIn("filelock==3.32.0", requirements)
        self.assertIn(
            "d396bea984af47333ef05e50eae7eff88c84256de6112aea0ec48a233c064fe3",
            requirements,
        )
        self.assertIn("importlib.util.find_spec(\"tokenizers\") is None", page)
        self.assertIn("enable_webgpu_rms_norm_fusion", page)
        self.assertIn("enable_webgpu_rotary_scaling_compatibility", page)
        self.assertIn("enable_webgpu_swiglu_fusion", page)
        self.assertIn("enable_webgpu_opt_sdpa_mask_compatibility", page)
        self.assertIn("enable_webgpu_preallocated_kv_cache", page)
        self.assertIn("WebGPUPreallocatedCache", page)
        self.assertIn("preallocated_kv_cache_probe", page)
        self.assertIn("convert_linear_modules_q8_", page)
        self.assertIn("q8_conversion_probe", page)
        self.assertIn("host_mask_truth_readbacks_avoided", page)
        self.assertIn('fused_calls == 1', page)
        self.assertIn('dispatches == 2', page)
        self.assertIn('upstream_dispatches - dispatches == 3', page)
        self.assertIn("identity_dispatches_saved_per_call", page)
        self.assertIn('fused_recorder.counts == {"aten::rms_norm": 1}', page)
        self.assertIn("upstream_operator_count == 6", page)
        self.assertIn("fused_dispatches == 1", page)
        self.assertIn('cpu_rms_norm_path"] = "unmodified_upstream_composition"', page)
        self.assertIn('fused_calls == 3', page)
        self.assertIn("failedModels.length !== 0", page)
        self.assertIn("TRANSFORMERS_RMS_NORM_ONLY", harness)
        self.assertIn('rms_norm_only: rmsNormOnly ? "1" : "0"', harness)
        self.assertIn("webgpuTestRmsNormOnly", page)
        self.assertIn('url.origin !== localOrigin', harness)
        self.assertIn(
            '"/runtime/transformers_browser_bootstrap.py"',
            harness,
        )
        self.assertIn('"/runtime/transformers_q8.py"', harness)
        self.assertIn('"/runtime/transformers_gemma2_webgpu.py"', harness)
        self.assertIn("tests/transformers-gemma2-webgpu.html", build)
        self.assertIn('"..",\n  "site",', harness)
        self.assertIn("scripts/prepare_transformers_browser_deps.py", build)
        self.assertNotIn('"filelock==3.32.0"', build)
        self.assertIn("Test hermetic Transformers WebGPU matrix", build)
        self.assertIn("node tests/webgpu-gemma2-wgsl.mjs", build)
        self.assertIn("node tests/webgpu-generation-long-wgsl.mjs", build)
        self.assertIn("node tests/transformers-webgpu.mjs", build)
        self.assertIn(
            '"Test hermetic Transformers WebGPU matrix"',
            publisher,
        )
        self.assertIn("scripts/prepare_transformers_browser_deps.py", pages)
        self.assertIn('staging / "runtime"', assembler)
        self.assertIn(
            'DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "transformers_tiny.json"',
            assembler,
        )
        playground = (ROOT / "tests" / "playground.mjs").read_text(
            encoding="utf-8"
        )
        self.assertIn('modeArgument === "--validate-only"', playground)
        self.assertIn("sha256File(wheelPath)", playground)
        self.assertIn("sha256File(filenamePath)", playground)
        self.assertIn("python_package_index_requests: 0", playground)
        smoke = (ROOT / "tests" / "smoke.mjs").read_text(encoding="utf-8")
        self.assertIn("the pinned filelock wheel is required", smoke)

    def test_transformers_browser_dependency_manifest_is_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheels = root / "wheels"
            wheels.mkdir()
            pins: list[tuple[str, str, str, str]] = []
            for requirement_name, metadata_name, version in (
                ("filelock", "filelock", "3.2.1"),
                ("huggingface-hub", "huggingface-hub", "1.2.3"),
                ("transformers", "transformers", "4.5.6"),
            ):
                filename_name = requirement_name.replace("-", "_")
                wheel = wheels / f"{filename_name}-{version}-py3-none-any.whl"
                dist_info = f"{filename_name}-{version}.dist-info"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr(
                        f"{dist_info}/METADATA",
                        "Metadata-Version: 2.1\n"
                        f"Name: {metadata_name}\n"
                        f"Version: {version}\n",
                    )
                    archive.writestr(
                        f"{dist_info}/WHEEL",
                        "Wheel-Version: 1.0\n"
                        "Root-Is-Purelib: true\n"
                        "Tag: py3-none-any\n",
                    )
                pins.append(
                    (
                        requirement_name,
                        version,
                        hashlib.sha256(wheel.read_bytes()).hexdigest(),
                        wheel.name,
                    )
                )

            requirements_path = root / "requirements.txt"
            requirements_path.write_text(
                "".join(
                    f"{name}=={version} --hash=sha256:{digest}\n"
                    for name, version, digest, _ in pins
                ),
                encoding="utf-8",
            )
            requirements = prepare_transformers_browser_deps.parse_requirements(
                requirements_path
            )
            packages = prepare_transformers_browser_deps.verify_wheels(
                requirements, wheels
            )
            manifest_path = prepare_transformers_browser_deps.write_manifest(
                packages, wheels
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["tokenizers_included"])
            self.assertEqual(
                [package["filename"] for package in manifest["packages"]],
                [pin[3] for pin in pins],
            )
            self.assertEqual(
                [package["sha256"] for package in manifest["packages"]],
                [pin[2] for pin in pins],
            )

    def test_playground_assembly_validates_and_copies_artifact_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / "site"
            dist = site / "dist"
            dist.mkdir(parents=True)
            for name in assemble_playground.SHELL_FILES:
                (site / name).write_text(name, encoding="utf-8")
            for name in assemble_playground.DIST_FILES:
                (dist / name).write_text(name, encoding="utf-8")

            release_dir = root / "release"
            release_dir.mkdir()
            wheel = release_dir / "torch-test.whl"
            wheel.write_bytes(b"tested wheel")
            wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            (release_dir / f"{wheel.name}.sha256").write_text(
                f"{wheel_digest}  {wheel.name}\n", encoding="utf-8"
            )
            configuration = config.load()
            (release_dir / "build-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "builder_repository_commit": None,
                        "configuration": configuration,
                        "inputs": {},
                        "wheel": {
                            "filename": wheel.name,
                            "sha256": wheel_digest,
                            "size": wheel.stat().st_size,
                        },
                    }
                ),
                encoding="utf-8",
            )

            transformers_dir = root / "transformers"
            transformers_dir.mkdir()
            pins: list[tuple[str, str, str]] = []
            for requirement_name, metadata_name, version in (
                ("filelock", "filelock", "3.2.1"),
                ("huggingface-hub", "huggingface-hub", "1.2.3"),
                ("transformers", "transformers", "4.5.6"),
            ):
                filename_name = requirement_name.replace("-", "_")
                dependency = (
                    transformers_dir
                    / f"{filename_name}-{version}-py3-none-any.whl"
                )
                dist_info = f"{filename_name}-{version}.dist-info"
                with zipfile.ZipFile(dependency, "w") as archive:
                    archive.writestr(
                        f"{dist_info}/METADATA",
                        "Metadata-Version: 2.1\n"
                        f"Name: {metadata_name}\n"
                        f"Version: {version}\n",
                    )
                    archive.writestr(
                        f"{dist_info}/WHEEL",
                        "Wheel-Version: 1.0\n"
                        "Root-Is-Purelib: true\n"
                        "Tag: py3-none-any\n",
                    )
                pins.append(
                    (
                        requirement_name,
                        version,
                        hashlib.sha256(dependency.read_bytes()).hexdigest(),
                    )
                )
            requirements_path = root / "requirements.txt"
            requirements_path.write_text(
                "".join(
                    f"{name}=={version} --hash=sha256:{digest}\n"
                    for name, version, digest in pins
                ),
                encoding="utf-8",
            )
            requirements = prepare_transformers_browser_deps.parse_requirements(
                requirements_path
            )
            packages = prepare_transformers_browser_deps.verify_wheels(
                requirements, transformers_dir
            )
            prepare_transformers_browser_deps.write_manifest(
                packages, transformers_dir
            )
            fixture = root / "transformers_tiny.json"
            fixture.write_text(
                json.dumps({"transformers_version": "4.5.6"}),
                encoding="utf-8",
            )
            output = root / "public"

            with mock.patch.object(
                prepare_transformers_browser_deps,
                "DEFAULT_REQUIREMENTS",
                requirements_path,
            ):
                result = assemble_playground.assemble(
                    release_dir,
                    transformers_dir,
                    output,
                    site_dir=site,
                    fixture=fixture,
                )

            self.assertEqual(
                result["release"], configuration["release"]["tag"]
            )
            self.assertEqual(result["transformers"], "4.5.6")
            self.assertEqual(
                (output / "runtime" / "build-manifest.json").read_bytes(),
                (release_dir / "build-manifest.json").read_bytes(),
            )
            self.assertTrue((output / "runtime" / wheel.name).is_file())
            self.assertTrue(
                (
                    output
                    / "runtime"
                    / "transformers"
                    / "transformers-browser-manifest.json"
                ).is_file()
            )
            with self.assertRaisesRegex(ValueError, "already exists"):
                assemble_playground.assemble(
                    release_dir,
                    transformers_dir,
                    output,
                    site_dir=site,
                    fixture=fixture,
                )

    def test_playground_assembly_rejects_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "filelock-1.0-py3-none-any.whl"
            dist_info = "filelock-1.0.dist-info"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    f"{dist_info}/METADATA",
                    "Metadata-Version: 2.1\nName: filelock\nVersion: 1.0\n",
                )
                archive.writestr(
                    f"{dist_info}/WHEEL",
                    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\n"
                    "Tag: py3-none-any\n",
                )
            requirements_path = root / "requirements.txt"
            requirements_path.write_text(
                "filelock==1.0 --hash=sha256:" + "0" * 64 + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                prepare_transformers_browser_deps,
                "DEFAULT_REQUIREMENTS",
                requirements_path,
            ):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    assemble_playground.validate_transformers_artifacts(root)

    def test_release_promotion_is_automatic_and_artifact_exact(self) -> None:
        workflows = ROOT / ".github" / "workflows"
        build = (workflows / "build.yml").read_text(encoding="utf-8")
        publisher = (workflows / "publish-release.yml").read_text(
            encoding="utf-8"
        )
        pages = (workflows / "pages.yml").read_text(encoding="utf-8")

        self.assertIn("Reserve release identity", build)
        self.assertIn('      - "site/**"', build)
        self.assertNotIn("gh release create", build)
        self.assertNotIn("\n  create:", build)

        self.assertIn("workflow_run:", publisher)
        self.assertIn("github.event.workflow_run.id", publisher)
        self.assertIn('source_workflow="${source_workflow%%@*}"', publisher)
        self.assertIn("Reverify latest successful build", publisher)
        self.assertIn("max_by(.id)", publisher)
        self.assertIn("Check out exact build source", publisher)
        self.assertIn('"Attest wheel provenance"', publisher)
        self.assertIn("Verify build artifact attestations", publisher)
        self.assertIn('--repo "$GITHUB_REPOSITORY"', publisher)
        self.assertIn(
            '--signer-workflow "$GITHUB_REPOSITORY/.github/workflows/build.yml"',
            publisher,
        )
        self.assertIn("--source-ref refs/heads/main", publisher)
        self.assertIn('--source-digest "$BUILDER_COMMIT"', publisher)
        self.assertIn("--deny-self-hosted-runners", publisher)
        self.assertNotIn("environment: release", publisher)
        self.assertIn("Publish release idempotently", publisher)
        self.assertIn("--draft", publisher)
        self.assertIn('gh release edit "$RELEASE_TAG" --draft=false', publisher)
        self.assertIn("Release asset inventory does not match", publisher)
        self.assertIn(
            "published-${{ github.run_id }}-${{ github.run_attempt }}",
            publisher,
        )
        self.assertNotIn("\n  create:", publisher)

        self.assertIn("Publish verified release artifact", pages)
        self.assertIn("Resolve newest complete Publisher run", pages)
        self.assertIn("publish-release.yml/runs", pages)
        self.assertIn('"Test hermetic Transformers WebGPU matrix"', publisher)
        self.assertIn('"Verify build artifact attestations"', pages)
        self.assertIn('"Upload promoted artifact for Pages"', pages)
        self.assertNotIn("releases/latest", pages)
        self.assertNotIn("workflows/build.yml/runs", pages)
        self.assertNotIn("\n  push:", pages)
        self.assertNotIn("\n      - Build PyTorch wheel", pages)
        self.assertIn(
            "published-${{ steps.publisher.outputs.run_id }}-"
            "${{ steps.publisher.outputs.run_attempt }}",
            pages,
        )
        self.assertIn("run-id: ${{ steps.publisher.outputs.run_id }}", pages)
        self.assertNotIn("gh release download", pages)
        self.assertIn("scripts/assemble_playground.py", pages)
        self.assertIn("node tests/playground.mjs public", pages)
        self.assertNotIn("release_args", pages)

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
        manifest["release"]["tag"] = "torch-2.13.0-pyodide-9.9.9-r5"
        self.assertIn(
            "release.tag must contain the pinned Pyodide version",
            config.validate(manifest),
        )

    def test_release_tag_and_wheel_revision_cannot_drift(self) -> None:
        manifest = config.load()
        manifest["pytorch"]["version"] = "2.13.0+pyodide314.0.2.r4"
        self.assertIn(
            "pytorch.version must encode the same torch, Pyodide, and "
            "release revision as release.tag",
            config.validate(manifest),
        )

    def test_release_tag_rejects_invalid_git_ref_components(self) -> None:
        manifest = config.load()
        manifest["release"]["tag"] = "torch-a..b-pyodide-314.0.2-r1"
        self.assertIn(
            "release.tag must match torch-X.Y.Z-pyodide-X.Y.Z-rN",
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
