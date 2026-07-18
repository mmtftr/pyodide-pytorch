import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const MAX_ERROR_CHARACTERS = 4_000;

let currentStage = "argument validation";
let wheel;
let wheelBytes;

function errorDetail(error) {
  const detail =
    error instanceof Error
      ? `${error.name}: ${error.message}`
      : String(error);
  if (detail.length <= MAX_ERROR_CHARACTERS) {
    return detail;
  }
  return `${detail.slice(0, MAX_ERROR_CHARACTERS)}… [truncated]`;
}

async function runStage(name, operation) {
  currentStage = name;
  const started = Date.now();
  console.log(`smoke: ${name}`);
  const result = await operation();
  console.log(`smoke: ${name} passed in ${Date.now() - started} ms`);
  return result;
}

async function main() {
  const [wheelArgument, expectedVersion, expectedCommit] = process.argv.slice(2);
  if (!wheelArgument || !expectedVersion || !expectedCommit) {
    throw new Error(
      "usage: node tests/smoke.mjs WHEEL EXPECTED_VERSION EXPECTED_COMMIT",
    );
  }

  wheel = path.resolve(wheelArgument);
  if (!fs.existsSync(wheel)) {
    throw new Error(`wheel does not exist: ${wheel}`);
  }
  wheelBytes = fs.statSync(wheel).size;
  console.log(
    JSON.stringify({
      smoke: "environment",
      node: process.version,
      wheel,
      wheel_bytes: wheelBytes,
    }),
  );

  const { loadPyodide } = await runStage(
    "load Pyodide JavaScript package",
    async () => require("pyodide"),
  );
  const pyodide = await runStage("initialize Pyodide", () => loadPyodide());

  await runStage("load runtime dependencies", () =>
    pyodide.loadPackage([
      "micropip",
      "numpy",
      "typing-extensions",
      "sympy",
      "networkx",
      "jinja2",
      "fsspec",
    ]),
  );
  await runStage("install filelock", () =>
    pyodide.runPythonAsync(`
import micropip
await micropip.install("filelock")
`),
  );
  await runStage("load torch wheel", () => pyodide.loadPackage(wheel));

  pyodide.globals.set("EXPECTED_TORCH_VERSION", expectedVersion);
  pyodide.globals.set("EXPECTED_TORCH_COMMIT", expectedCommit);
  const result = await runStage("run torch assertions", () =>
    pyodide.runPythonAsync(`
import io
import json
import sys

import torch

assert sys.platform == "emscripten", sys.platform
assert torch.__version__ == EXPECTED_TORCH_VERSION, torch.__version__
assert torch.version.git_version == EXPECTED_TORCH_COMMIT, torch.version.git_version
assert not torch.cuda.is_available()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1

for setter in (torch.set_num_threads, torch.set_num_interop_threads):
    try:
        setter(2)
    except RuntimeError as error:
        assert "single-threaded" in str(error)
    else:
        raise AssertionError(f"{setter.__name__}(2) unexpectedly succeeded")

for setter in (torch.set_num_threads, torch.set_num_interop_threads):
    try:
        setter(2**65)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"{setter.__name__}(2**65) unexpectedly succeeded")

assert torch.get_num_threads() == 1
assert torch.get_num_interop_threads() == 1

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
loss = (x @ x).sum()
loss.backward()
torch.testing.assert_close(
    x.grad,
    torch.tensor([[7.0, 11.0], [9.0, 13.0]]),
)

model = torch.nn.Linear(2, 1)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[0.25, -0.5]]))
    model.bias.zero_()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
features = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
targets = torch.tensor([[1.0], [0.0]])
before = torch.nn.functional.mse_loss(model(features), targets)
optimizer.zero_grad()
before.backward()
optimizer.step()
after = torch.nn.functional.mse_loss(model(features), targets)
assert after < before, (before.item(), after.item())

buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
buffer.seek(0)
restored = torch.load(buffer)
assert set(restored) == {"weight", "bias"}
torch.testing.assert_close(restored["weight"], model.weight)

gradient = torch.func.grad(lambda value: value.square().sum())(
    torch.tensor([2.0, -3.0])
)
torch.testing.assert_close(gradient, torch.tensor([4.0, -6.0]))

json.dumps({
    "torch": torch.__version__,
    "git": torch.version.git_version,
    "intraop_threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "autograd": True,
    "optimizer": True,
    "serialization": True,
    "torch_func": True,
})
`),
  );

  console.log(JSON.stringify(JSON.parse(result), null, 2));
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        smoke: "failed",
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
