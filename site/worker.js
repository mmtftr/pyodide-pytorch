let pyodide = null;
let runtimeReady = false;
let running = false;
let taskQueue = Promise.resolve();

function postStatus(stage, label, detail) {
  self.postMessage({ type: "status", stage, label, detail });
}

function errorText(error) {
  if (error instanceof Error) return error.stack || error.message;
  return String(error);
}

async function initialize(config) {
  try {
    postStatus("pyodide", "Loading Pyodide", `Runtime ${config.pyodideVersion}.`);
    const indexURL = `https://cdn.jsdelivr.net/pyodide/v${config.pyodideVersion}/full/`;
    const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
    pyodide = await loadPyodide({ indexURL });

    pyodide.setStdout({ batched: (text) => self.postMessage({ type: "stdout", text }) });
    pyodide.setStderr({ batched: (text) => self.postMessage({ type: "stderr", text }) });

    postStatus("dependencies", "Loading Python dependencies", "NumPy, SymPy, NetworkX, Jinja2, and FSSpec.");
    await pyodide.loadPackage([
      "micropip",
      "numpy",
      "typing-extensions",
      "sympy",
      "networkx",
      "jinja2",
      "fsspec",
    ]);

    postStatus("filelock", "Installing filelock", "Installing the pure-Python dependency.");
    await pyodide.runPythonAsync(`
import micropip
await micropip.install("filelock")
`);

    postStatus("torch", "Loading PyTorch wheel", config.wheelName);
    await pyodide.loadPackage(config.wheelUrl);

    postStatus("verify", "Verifying runtime", "Checking version, platform, and thread invariants.");
    const detailsJson = await pyodide.runPythonAsync(`
import json
import sys
import torch

json.dumps({
    "version": torch.__version__,
    "platform": sys.platform,
    "intra_threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
})
`);
    const details = JSON.parse(detailsJson);
    if (details.version !== config.torchVersion) {
      throw new Error(`Expected torch ${config.torchVersion}, loaded ${details.version}.`);
    }
    if (details.platform !== "emscripten") {
      throw new Error(`Expected the Emscripten platform, loaded ${details.platform}.`);
    }
    if (details.intra_threads !== 1 || details.interop_threads !== 1) {
      throw new Error("The runtime violated its single-threaded build invariant.");
    }

    runtimeReady = true;
    self.postMessage({ type: "ready", details });
  } catch (error) {
    self.postMessage({ type: "fatal", error: errorText(error) });
  }
}

async function run(code) {
  if (!runtimeReady || running) return;
  running = true;
  self.postMessage({ type: "run-started" });
  let result;
  try {
    result = await pyodide.runPythonAsync(String(code));
    if (result !== undefined && result !== null) {
      self.postMessage({ type: "result", text: String(result) });
    }
  } catch (error) {
    self.postMessage({ type: "stderr", text: errorText(error) });
  } finally {
    result?.destroy?.();
    running = false;
    self.postMessage({ type: "run-finished" });
  }
}

async function inspectExpression(expression) {
  if (!runtimeReady || running) return null;
  const resultJson = await pyodide.runPythonAsync(`
import builtins
import inspect
import json
import re

_expression = ${JSON.stringify(expression)}
_result = None

if re.fullmatch(r"[A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*", _expression):
    try:
        _object = eval(_expression, globals(), vars(builtins))
    except Exception:
        _object = None

    if _object is not None:
        try:
            _signature = str(inspect.signature(_object))
        except (TypeError, ValueError):
            _signature = ""

        try:
            _documentation = inspect.getdoc(_object) or ""
        except Exception:
            _documentation = ""

        _result = {
            "symbol": _expression,
            "signature": _expression + _signature if _signature else _expression,
            "documentation": _documentation[:4000],
            "module": getattr(_object, "__module__", "") or "",
            "qualname": getattr(_object, "__qualname__", "") or getattr(_object, "__name__", "") or "",
        }

json.dumps(_result)
`);
  return JSON.parse(resultJson);
}

async function completeExpression(expression, prefix) {
  if (!runtimeReady || running) return [];
  const resultJson = await pyodide.runPythonAsync(`
import builtins
import inspect
import json
import re

_expression = ${JSON.stringify(expression)}
_prefix = ${JSON.stringify(prefix)}
_results = []

if (
    re.fullmatch(r"[A-Za-z_]\\w*(?:\\.[A-Za-z_]\\w*)*", _expression)
    and re.fullmatch(r"\\w*", _prefix)
):
    try:
        _object = eval(_expression, globals(), vars(builtins))
    except Exception:
        _object = None

    if _object is not None:
        for _name in (name for name in dir(_object) if not name.startswith("_") and name.startswith(_prefix)):
            if len(_results) >= 200:
                break
            try:
                _member = getattr(_object, _name)
            except Exception:
                continue

            if inspect.ismodule(_member):
                _kind = "module"
            elif inspect.isclass(_member):
                _kind = "class"
            elif callable(_member):
                _kind = "function"
            else:
                _kind = "variable"

            try:
                _signature = str(inspect.signature(_member)) if callable(_member) else ""
            except (TypeError, ValueError):
                _signature = ""

            try:
                _documentation = (inspect.getdoc(_member) or "").split("\\n\\n", 1)[0]
            except Exception:
                _documentation = ""

            _results.append({
                "name": _name,
                "kind": _kind,
                "signature": _name + _signature if _signature else "",
                "documentation": _documentation[:600],
                "module": getattr(_member, "__module__", "") or "",
            })

json.dumps(_results)
`);
  return JSON.parse(resultJson);
}

async function handleRequest(message) {
  let result = null;
  try {
    if (message.type === "inspect") {
      result = await inspectExpression(String(message.expression ?? ""));
    } else if (message.type === "complete") {
      result = await completeExpression(
        String(message.expression ?? ""),
        String(message.prefix ?? ""),
      );
    }
  } catch (error) {
    console.warn("Runtime inspection failed", error);
  }
  self.postMessage({ type: "request-result", id: message.id, result });
}

function enqueue(task) {
  taskQueue = taskQueue.then(task).catch((error) => {
    console.error("Pyodide worker task failed", error);
  });
}

self.addEventListener("message", (event) => {
  const message = event.data ?? {};
  if (message.type === "init" && !pyodide) {
    enqueue(() => initialize(message.config));
  } else if (message.type === "run") {
    enqueue(() => run(message.code));
  } else if ((message.type === "inspect" || message.type === "complete") && message.id) {
    enqueue(() => handleRequest(message));
  }
});
