from __future__ import annotations

import ast
import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "patches"
    / "pytorch"
    / "0010-add-experimental-browser-webgpu-backend.patch"
)


def _added_file_source(patch: str, path: str) -> str:
    marker = f"diff --git a/{path} b/{path}\n"
    section = patch.split(marker, 1)[1].split("\ndiff --git ", 1)[0]
    lines = section.splitlines()
    hunk = next(index for index, line in enumerate(lines) if line.startswith("@@ "))
    return "\n".join(
        line[1:] for line in lines[hunk + 1 :] if line.startswith("+")
    ) + "\n"


class StridedReadbackContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patch = PATCH.read_text(encoding="utf-8")
        cls.webgpu_python = _added_file_source(
            cls.patch, "torch/webgpu/__init__.py"
        )
        cls.python_tree = ast.parse(
            cls.webgpu_python, filename="torch/webgpu/__init__.py"
        )
        cls.to_cpu_node = next(
            node
            for node in cls.python_tree.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "to_cpu_async"
        )
        lines = cls.webgpu_python.splitlines()
        cls.to_cpu_source = "\n".join(
            lines[cls.to_cpu_node.lineno - 1 : cls.to_cpu_node.end_lineno]
        )
        cls.copy_cpp = (
            ROOT / "webgpu" / "llm_kernels" / "copy.cpp"
        ).read_text(encoding="utf-8")
        cls.functional_gate = (ROOT / "tests" / "webgpu.html").read_text(
            encoding="utf-8"
        )
        cls.transformers_gate = (
            ROOT / "tests" / "transformers-webgpu.html"
        ).read_text(encoding="utf-8")
        cls.raw_gate = (
            ROOT / "tests" / "webgpu-bool-wgsl.mjs"
        ).read_text(encoding="utf-8")

    def test_wrapper_materializes_before_the_single_async_readback(self) -> None:
        source = self.to_cpu_source
        materialize = source.index("tensor = tensor.contiguous()")
        flush = source.index("_flush()")
        identify = source.index("torch.ops.webgpu.buffer_id(tensor)")
        map_async = source.index("await readback.mapAsync")
        self.assertLess(materialize, flush)
        self.assertLess(flush, identify)
        self.assertLess(identify, map_async)
        self.assertEqual(source.count("await readback.mapAsync"), 1)
        self.assertEqual(source.count("np.frombuffer("), 1)
        self.assertNotIn("readback requires a contiguous tensor", source)
        self.assertNotIn(".cpu()", source)

    def test_noncontiguous_empty_view_uses_materialized_tensor(self) -> None:
        events: list[object] = []
        float32 = object()
        int32 = object()
        int64 = object()
        boolean = object()

        class Tensor:
            pass

        class FakeTensor(Tensor):
            def __init__(self, *, contiguous: bool) -> None:
                self.device = types.SimpleNamespace(type="webgpu")
                self.dtype = float32
                self.shape = (2, 0, 3)
                self._contiguous = contiguous
                self.materialized: FakeTensor | None = None

            def is_contiguous(self) -> bool:
                return self._contiguous

            def contiguous(self) -> FakeTensor:
                events.append("contiguous")
                self.materialized = FakeTensor(contiguous=True)
                return self.materialized

            def numel(self) -> int:
                return 0

        original = FakeTensor(contiguous=False)

        def buffer_nbytes(tensor: FakeTensor) -> int:
            events.append(("buffer_nbytes", tensor))
            return 0

        def empty(shape: tuple[int, ...], *, dtype: object) -> object:
            result = (shape, dtype)
            events.append(("empty", result))
            return result

        fake_torch = types.SimpleNamespace(
            Tensor=Tensor,
            float32=float32,
            int32=int32,
            long=int64,
            bool=boolean,
            empty=empty,
            ops=types.SimpleNamespace(
                webgpu=types.SimpleNamespace(buffer_nbytes=buffer_nbytes)
            ),
        )
        function = ast.Module(body=[self.to_cpu_node], type_ignores=[])
        ast.fix_missing_locations(function)
        namespace = {
            "torch": fake_torch,
            "_flush": lambda: events.append("flush"),
        }
        exec(compile(function, "torch/webgpu/__init__.py", "exec"), namespace)
        fake_numpy = types.ModuleType("numpy")
        with mock.patch.dict(sys.modules, {"numpy": fake_numpy}):
            result = asyncio.run(namespace["to_cpu_async"](original))

        self.assertIsNotNone(original.materialized)
        self.assertEqual(
            events[:3],
            [
                "contiguous",
                "flush",
                ("buffer_nbytes", original.materialized),
            ],
        )
        self.assertEqual(result, ((2, 0, 3), float32))

    def test_existing_gpu_copy_registration_is_reused_once(self) -> None:
        self.assertEqual(self.copy_cpp.count('module.impl("contiguous"'), 1)
        self.assertIn(
            "return clone_impl(self, c10::MemoryFormat::Contiguous);",
            self.copy_cpp,
        )
        self.assertIn("copy_strided(self, output);", self.copy_cpp)
        self.assertIn("source.storage_offset()", self.copy_cpp)
        self.assertIn("source.stride(dim)", self.copy_cpp)
        self.assertNotIn("at::kCPU", self.copy_cpp)

    def test_browser_gate_reads_views_without_test_side_materialization(self) -> None:
        for name in (
            "bool_strided_view",
            "long_strided_view",
            "strided",
            "transposed",
            "readback_prefix",
        ):
            self.assertIn(
                f"await torch.webgpu.to_cpu_async({name})",
                self.functional_gate,
            )
            self.assertNotIn(
                f"await torch.webgpu.to_cpu_async({name}.contiguous())",
                self.functional_gate,
            )
        self.assertIn(
            "assert not readback_prefix.is_contiguous()",
            self.functional_gate,
        )
        self.assertIn(
            "assert not cache_keys_gpu.is_contiguous()",
            self.transformers_gate,
        )
        self.assertIn(
            "cache_keys_gpu.cpu()",
            self.transformers_gate,
        )
        self.assertNotIn("cache_keys_gpu.contiguous().cpu()", self.transformers_gate)

    def test_raw_shader_gate_covers_offset_prefix_and_output_bounds(self) -> None:
        self.assertIn('"strided_copy.wgsl"', self.raw_gate)
        self.assertIn("strided copy offset prefix source", self.raw_gate)
        self.assertIn("prefixParams[2] = 64", self.raw_gate)
        self.assertIn("prefixParams.set([60, 20, 4, 1], 16)", self.raw_gate)
        self.assertIn("new Array(4).fill(-777.5)", self.raw_gate)
        self.assertIn("strided copy offset prefix mismatch", self.raw_gate)


if __name__ == "__main__":
    unittest.main()
