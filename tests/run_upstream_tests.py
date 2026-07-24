from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import sys
import time
import traceback
import types
import unittest
from collections.abc import Iterator
from difflib import get_close_matches
from pathlib import Path
from typing import Any


class JsonTestResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.passed_count = 0
        self.skipped_details: list[dict[str, Any]] = []
        self.expected_failure_details: list[dict[str, Any]] = []
        self.unexpected_success_details: list[dict[str, Any]] = []
        self.failure_details: list[dict[str, Any]] = []
        self.error_details: list[dict[str, Any]] = []
        self._started_at: dict[str, float] = {}

    @staticmethod
    def _test_id(test: unittest.case.TestCase) -> str:
        return test.id()

    def startTest(self, test: unittest.case.TestCase) -> None:
        super().startTest(test)
        test_id = self._test_id(test)
        self._started_at[test_id] = time.perf_counter()
        print(f"upstream: RUN  {test_id}", flush=True)

    def _duration_ms(self, test: unittest.case.TestCase) -> int:
        started = self._started_at.pop(self._test_id(test), time.perf_counter())
        return round((time.perf_counter() - started) * 1_000)

    def addSuccess(self, test: unittest.case.TestCase) -> None:
        super().addSuccess(test)
        test_id = self._test_id(test)
        duration_ms = self._duration_ms(test)
        self.passed_count += 1
        print(f"upstream: PASS {test_id} ({duration_ms} ms)", flush=True)

    def addSkip(self, test: unittest.case.TestCase, reason: str) -> None:
        super().addSkip(test, reason)
        test_id = self._test_id(test)
        duration_ms = self._duration_ms(test)
        self.skipped_details.append(
            {"id": test_id, "reason": reason, "duration_ms": duration_ms}
        )
        print(f"upstream: SKIP {test_id}: {reason}", flush=True)

    def addFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        super().addFailure(test, err)
        test_id = self._test_id(test)
        duration_ms = self._duration_ms(test)
        detail = "".join(traceback.format_exception(*err))
        self.failure_details.append(
            {"id": test_id, "traceback": detail, "duration_ms": duration_ms}
        )
        print(f"upstream: FAIL {test_id} ({duration_ms} ms)", flush=True)

    def addError(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        super().addError(test, err)
        test_id = self._test_id(test)
        duration_ms = self._duration_ms(test)
        detail = "".join(traceback.format_exception(*err))
        self.error_details.append(
            {"id": test_id, "traceback": detail, "duration_ms": duration_ms}
        )
        print(f"upstream: ERROR {test_id} ({duration_ms} ms)", flush=True)

    def addExpectedFailure(
        self,
        test: unittest.case.TestCase,
        err: tuple[type[BaseException], BaseException, Any],
    ) -> None:
        super().addExpectedFailure(test, err)
        test_id = self._test_id(test)
        duration_ms = self._duration_ms(test)
        detail = "".join(traceback.format_exception(*err))
        self.expected_failure_details.append(
            {"id": test_id, "traceback": detail, "duration_ms": duration_ms}
        )
        print(f"upstream: XFAIL {test_id} ({duration_ms} ms)", flush=True)

    def addUnexpectedSuccess(self, test: unittest.case.TestCase) -> None:
        super().addUnexpectedSuccess(test)
        test_id = self._test_id(test)
        duration_ms = self._duration_ms(test)
        self.unexpected_success_details.append(
            {"id": test_id, "duration_ms": duration_ms}
        )
        print(f"upstream: XPASS {test_id} ({duration_ms} ms)", flush=True)


def load_upstream_module(root: Path, relative_path: str, index: int) -> Any:
    root = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"upstream test path escapes the source root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"upstream test source is missing: {path}")
    module_name = f"_pytorch_upstream_{index}_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def iter_tests(suite: unittest.TestSuite) -> Iterator[unittest.case.TestCase]:
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from iter_tests(test)
        else:
            yield test


def relative_test_id(test: unittest.case.TestCase, module: Any) -> str:
    prefix = f"{module.__name__}."
    test_id = test.id()
    if not test_id.startswith(prefix):
        raise ValueError(f"unexpected test ID outside {module.__name__}: {test_id}")
    return test_id[len(prefix) :]


def validate_config(config_json: str) -> dict[str, Any]:
    config = json.loads(config_json)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported upstream test manifest schema")
    if not isinstance(config.get("modules"), list) or not config["modules"]:
        raise ValueError("upstream test manifest must contain at least one module")
    return config


def configure_test_process() -> None:
    os.environ.setdefault("PYTORCH_TEST_WITH_DYNAMO", "0")
    os.environ.setdefault("PYTORCH_TEST_WITH_SLOW", "0")
    os.environ.setdefault("PYTORCH_TEST_WITH_CROSSREF", "0")
    sys.argv = ["pytorch-wasm-upstream-tests"]


def install_collection_stubs(config: dict[str, Any]) -> list[dict[str, str]]:
    installed = []
    stubbed_names = set()
    for stub_config in config.get("collection_stubs", []):
        module_name = stub_config["module"]
        reason = stub_config["reason"]
        if not module_name or any(
            not part.isidentifier() for part in module_name.split(".")
        ):
            raise ValueError(f"invalid collection stub module: {module_name!r}")
        if not reason.strip():
            raise ValueError(f"empty collection stub reason for {module_name}")
        if module_name in stubbed_names:
            raise ValueError(f"duplicate collection stub: {module_name}")
        stubbed_names.add(module_name)

        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing_name = error.name or ""
            if missing_name != module_name and not module_name.startswith(
                f"{missing_name}."
            ):
                raise
        else:
            raise ValueError(
                f"collection stub target is importable and must not be hidden: "
                f"{module_name}"
            )

        parent = None
        parts = module_name.split(".")
        for index, part in enumerate(parts):
            qualified_name = ".".join(parts[: index + 1])
            module = sys.modules.get(qualified_name)
            if module is None:
                module = types.ModuleType(qualified_name)
                if index < len(parts) - 1:
                    module.__path__ = []  # type: ignore[attr-defined]
                sys.modules[qualified_name] = module
            if parent is not None:
                setattr(parent, part, module)
            parent = module
        installed.append({"module": module_name, "reason": reason})
    return installed


def discover_manifest(config_json: str, root: str) -> str:
    config = validate_config(config_json)
    configure_test_process()
    collection_stubs = install_collection_stubs(config)
    loader = unittest.TestLoader()
    discovered_modules = []
    for index, module_config in enumerate(config["modules"]):
        module = load_upstream_module(Path(root), module_config["path"], index)
        tests = list(iter_tests(loader.loadTestsFromModule(module)))
        discovered_modules.append(
            {
                "path": module_config["path"],
                "count": len(tests),
                "tests": [relative_test_id(test, module) for test in tests],
            }
        )
    return json.dumps(
        {
            "schema_version": 1,
            "collection_stubs": collection_stubs,
            "modules": discovered_modules,
            "total": sum(module["count"] for module in discovered_modules),
        },
        indent=2,
        sort_keys=True,
    )


def load_manifest_suite(
    config: dict[str, Any], root: Path
) -> tuple[unittest.TestSuite, list[dict[str, str]]]:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    excluded_tests: list[dict[str, str]] = []
    for index, module_config in enumerate(config["modules"]):
        module = load_upstream_module(root, module_config["path"], index)
        discovered = list(iter_tests(loader.loadTestsFromModule(module)))
        tests_by_id = {relative_test_id(test, module): test for test in discovered}
        if len(tests_by_id) != len(discovered):
            raise ValueError(f"duplicate generated test IDs in {module_config['path']}")
        if module_config["tests"] == ["*"]:
            exclusions = module_config.get("excluded_tests", [])
            excluded_ids = set()
            for exclusion in exclusions:
                test_id = exclusion["id"]
                reason = exclusion["reason"]
                if not reason.strip():
                    raise ValueError(f"empty exclusion reason for {test_id}")
                if test_id not in tests_by_id:
                    raise ValueError(
                        f"excluded test is not present in {module_config['path']}: "
                        f"{test_id}"
                    )
                if test_id in excluded_ids:
                    raise ValueError(f"duplicate excluded test: {test_id}")
                excluded_ids.add(test_id)
                excluded_tests.append(
                    {
                        "source": module_config["path"],
                        "id": test_id,
                        "reason": reason,
                    }
                )
            suite.addTests(
                test
                for test_id, test in tests_by_id.items()
                if test_id not in excluded_ids
            )
            continue
        if module_config.get("excluded_tests"):
            raise ValueError(
                "excluded_tests is only supported for whole-module test selection"
            )
        selected_ids = set()
        for test_name in module_config["tests"]:
            if test_name in selected_ids:
                raise ValueError(
                    f"duplicate selected test in {module_config['path']}: {test_name}"
                )
            selected_ids.add(test_name)
            if test_name not in tests_by_id:
                suggestions = get_close_matches(test_name, tests_by_id, n=4)
                suggestion_text = (
                    f"; closest generated IDs: {', '.join(suggestions)}"
                    if suggestions
                    else ""
                )
                raise ValueError(
                    f"selected test is not present in {module_config['path']}: "
                    f"{test_name}{suggestion_text}"
                )
            suite.addTest(tests_by_id[test_name])
    return suite, excluded_tests


def run_manifest(config_json: str, root: str) -> str:
    config = validate_config(config_json)
    configure_test_process()
    collection_stubs = install_collection_stubs(config)

    random.seed(0)
    import numpy as np
    import torch

    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    started = time.perf_counter()
    suite, excluded_tests = load_manifest_suite(config, Path(root))
    result = JsonTestResult()
    suite.run(result)
    duration_ms = round((time.perf_counter() - started) * 1_000)

    summary = {
        "schema_version": 1,
        "torch": torch.__version__,
        "git": torch.version.git_version,
        "platform": sys.platform,
        "duration_ms": duration_ms,
        "total": result.testsRun,
        "passed": result.passed_count,
        "skipped": len(result.skipped_details),
        "excluded": len(excluded_tests),
        "collection_stubs": len(collection_stubs),
        "failures": len(result.failure_details),
        "errors": len(result.error_details),
        "expected_failures": len(result.expected_failure_details),
        "unexpected_successes": len(result.unexpectedSuccesses),
        "skipped_tests": result.skipped_details,
        "excluded_tests": excluded_tests,
        "collection_stub_details": collection_stubs,
        "failure_details": result.failure_details,
        "error_details": result.error_details,
        "expected_failure_details": result.expected_failure_details,
        "unexpected_success_details": result.unexpected_success_details,
    }
    return json.dumps(summary, indent=2, sort_keys=True)
