import childProcess from "node:child_process";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const MAX_ERROR_CHARACTERS = 8_000;

let currentStage = "argument validation";
let wheel;
let wheelBytes;

function errorDetail(error) {
  const detail =
    error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  if (detail.length <= MAX_ERROR_CHARACTERS) {
    return detail;
  }
  return `${detail.slice(0, MAX_ERROR_CHARACTERS)}… [truncated]`;
}

async function runStage(name, operation) {
  currentStage = name;
  const started = Date.now();
  console.log(`upstream: ${name}`);
  const result = await operation();
  console.log(`upstream: ${name} passed in ${Date.now() - started} ms`);
  return result;
}

function requireFile(file, description) {
  if (!fs.existsSync(file) || !fs.statSync(file).isFile()) {
    throw new Error(`${description} does not exist: ${file}`);
  }
}

function writePyodideFile(pyodide, destination, source) {
  pyodide.FS.mkdirTree(path.posix.dirname(destination));
  pyodide.FS.writeFile(destination, fs.readFileSync(source));
}

async function main() {
  const arguments_ = process.argv.slice(2);
  const listTestsIndex = arguments_.indexOf("--list-tests");
  const listTests = listTestsIndex !== -1;
  if (listTests) {
    arguments_.splice(listTestsIndex, 1);
  }
  const [
    wheelArgument,
    sourceArgument,
    expectedVersion,
    expectedCommit,
    manifestArgument = "tests/upstream_cpu_wasm.json",
  ] = arguments_;
  if (
    !wheelArgument ||
    !sourceArgument ||
    !expectedVersion ||
    !expectedCommit ||
    arguments_.length > 5
  ) {
    throw new Error(
      "usage: node tests/upstream.mjs WHEEL PYTORCH_SOURCE " +
        "EXPECTED_VERSION EXPECTED_COMMIT [MANIFEST] [--list-tests]",
    );
  }

  wheel = path.resolve(wheelArgument);
  const sourceRoot = path.resolve(sourceArgument);
  const manifestPath = path.resolve(manifestArgument);
  const harnessPath = path.resolve("tests/run_upstream_tests.py");
  requireFile(wheel, "wheel");
  requireFile(manifestPath, "upstream test manifest");
  requireFile(harnessPath, "upstream Python harness");
  if (!fs.existsSync(sourceRoot) || !fs.statSync(sourceRoot).isDirectory()) {
    throw new Error(`PyTorch source directory does not exist: ${sourceRoot}`);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  if (manifest.schema_version !== 1) {
    throw new Error(`unsupported manifest schema: ${manifest.schema_version}`);
  }
  if (manifest.pytorch_ref !== expectedCommit) {
    throw new Error(
      `manifest pins ${manifest.pytorch_ref}, expected ${expectedCommit}`,
    );
  }
  const sourceCommit = childProcess
    .execFileSync("git", ["-C", sourceRoot, "rev-parse", "HEAD"], {
      encoding: "utf8",
    })
    .trim();
  if (sourceCommit !== expectedCommit) {
    throw new Error(
      `PyTorch source is ${sourceCommit}, expected ${expectedCommit}`,
    );
  }
  const upstreamTestPaths = manifest.modules.map((module) => module.path);
  try {
    childProcess.execFileSync(
      "git",
      [
        "-C",
        sourceRoot,
        "diff",
        "--quiet",
        expectedCommit,
        "--",
        ...upstreamTestPaths,
      ],
      { stdio: "ignore" },
    );
  } catch (error) {
    if (error?.status === 1) {
      throw new Error(
        "selected upstream test files differ from the pinned PyTorch commit",
      );
    }
    throw error;
  }

  wheelBytes = fs.statSync(wheel).size;
  console.log(
    JSON.stringify({
      upstream: "environment",
      node: process.version,
      wheel,
      wheel_bytes: wheelBytes,
      pytorch_source: sourceRoot,
      pytorch_ref: sourceCommit,
      manifest: manifestPath,
      list_tests: listTests,
    }),
  );

  const { loadPyodide } = await runStage(
    "load Pyodide JavaScript package",
    async () => require("pyodide"),
  );
  const pyodide = await runStage("initialize Pyodide", () => loadPyodide());

  await runStage("load runtime dependencies", () =>
    pyodide.loadPackage(manifest.pyodide_packages),
  );
  await runStage("install upstream test dependencies", () => {
    pyodide.globals.set(
      "UPSTREAM_PYPI_PACKAGES",
      pyodide.toPy(manifest.pypi_packages),
    );
    return pyodide.runPythonAsync(`
import micropip
await micropip.install(UPSTREAM_PYPI_PACKAGES)
`);
  });
  await runStage("load torch wheel", () => pyodide.loadPackage(wheel));

  await runStage("copy pinned upstream tests", async () => {
    for (const module of manifest.modules) {
      const source = path.resolve(sourceRoot, module.path);
      const relativeSource = path.relative(sourceRoot, source);
      if (
        relativeSource === "" ||
        relativeSource.startsWith(`..${path.sep}`) ||
        path.isAbsolute(relativeSource)
      ) {
        throw new Error(
          `upstream test path escapes the source root: ${module.path}`,
        );
      }
      requireFile(source, "upstream test source");
      const destination = path.posix.resolve(
        "/pytorch-upstream",
        module.path,
      );
      if (!destination.startsWith("/pytorch-upstream/")) {
        throw new Error(
          `upstream test destination escapes its root: ${module.path}`,
        );
      }
      writePyodideFile(pyodide, destination, source);
    }
    writePyodideFile(
      pyodide,
      "/pytorch-upstream-harness/run_upstream_tests.py",
      harnessPath,
    );
  });

  pyodide.globals.set("UPSTREAM_CONFIG_JSON", JSON.stringify(manifest));
  pyodide.globals.set("UPSTREAM_EXPECTED_VERSION", expectedVersion);
  pyodide.globals.set("UPSTREAM_EXPECTED_COMMIT", expectedCommit);
  if (listTests) {
    const discoveryJson = await runStage("discover pinned upstream tests", () =>
      pyodide.runPythonAsync(`
import sys

sys.path.insert(0, "/pytorch-upstream-harness")
from run_upstream_tests import discover_manifest

discover_manifest(UPSTREAM_CONFIG_JSON, "/pytorch-upstream")
`),
    );
    console.log(discoveryJson);
    return;
  }

  const summaryJson = await runStage("run pinned upstream tests", () =>
    pyodide.runPythonAsync(`
import json
import sys

sys.path.insert(0, "/pytorch-upstream-harness")
from run_upstream_tests import run_manifest

summary_json = run_manifest(UPSTREAM_CONFIG_JSON, "/pytorch-upstream")
summary = json.loads(summary_json)
assert summary["torch"] == UPSTREAM_EXPECTED_VERSION, summary["torch"]
assert summary["git"] == UPSTREAM_EXPECTED_COMMIT, summary["git"]
assert summary["platform"] == "emscripten", summary["platform"]
summary_json
`),
  );

  const summary = JSON.parse(summaryJson);
  console.log(summaryJson);
  const expected = manifest.expected;
  const mismatches = [];
  if (expected) {
    for (const key of [
      "total",
      "passed",
      "skipped",
      "excluded",
      "collection_stubs",
      "expected_failures",
    ]) {
      if (summary[key] !== expected[key]) {
        mismatches.push(
          `${key}: expected ${expected[key]}, found ${summary[key]}`,
        );
      }
    }
  }
  for (const key of ["failures", "errors", "unexpected_successes"]) {
    if (summary[key] !== 0) {
      mismatches.push(`${key}: expected 0, found ${summary[key]}`);
    }
  }
  if (mismatches.length) {
    throw new Error(`upstream test contract failed: ${mismatches.join("; ")}`);
  }
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        upstream: "failed",
        stage: currentStage,
        node: process.version,
        wheel,
        wheel_bytes: wheelBytes,
        rss_bytes: process.memoryUsage().rss,
        error: errorDetail(error),
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
});
