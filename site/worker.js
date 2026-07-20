let pyodide = null;
let runtimeReady = false;
let running = false;

function postStatus(stage, label, detail) {
  self.postMessage({ type: "status", stage, label, detail });
}

function errorText(error) {
  if (error instanceof Error) return error.stack || error.message;
  return String(error);
}

async function initialize(config) {
  try {
    postStatus("pyodide", "Loading Pyodide", `Fetching runtime ${config.pyodideVersion}.`);
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

    postStatus("filelock", "Installing filelock", "Resolving the final pure-Python dependency.");
    await pyodide.runPythonAsync(`
import micropip
await micropip.install("filelock")
`);

    postStatus("torch", "Loading PyTorch wheel", `${config.wheelName} · this is the large download.`);
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

self.addEventListener("message", (event) => {
  const message = event.data ?? {};
  if (message.type === "init" && !pyodide) {
    initialize(message.config);
  } else if (message.type === "run") {
    run(message.code);
  }
});
