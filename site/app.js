import { completeFromList } from "@codemirror/autocomplete";
import { indentWithTab } from "@codemirror/commands";
import { python, pythonLanguage } from "@codemirror/lang-python";
import { EditorState, Prec } from "@codemirror/state";
import { oneDark } from "@codemirror/theme-one-dark";
import { keymap } from "@codemirror/view";
import { basicSetup, EditorView } from "codemirror";

const REPOSITORY = "mmtftr/pyodide-pytorch";
const RUNTIME_BASE_URL = new URL("./runtime/", document.baseURI);
const PUBLISHED_MANIFEST_URL = new URL("build-manifest.json", RUNTIME_BASE_URL);

const FALLBACK_RELEASE = Object.freeze({
  releaseTag: "torch-2.13.0-pyodide-314.0.2-r1",
  releaseUrl:
    "https://github.com/mmtftr/pyodide-pytorch/releases/tag/torch-2.13.0-pyodide-314.0.2-r1",
  wheelUrl: new URL(
    "torch-2.13.0+pyodide314.0.2-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
    RUNTIME_BASE_URL,
  ).href,
  wheelName:
    "torch-2.13.0+pyodide314.0.2-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
  wheelSize: 25_038_799,
  wheelSha256: "8691f0276528a7deee66c3abae3d21824ff8f2d20c9173142957bf04334af2a3",
  pyodideVersion: "314.0.2",
  torchVersion: "2.13.0+pyodide314.0.2",
});

const EXAMPLES = Object.freeze({
  autograd: {
    filename: "autograd.py",
    code: `import torch

print("torch:", torch.__version__)
print(
    "threads:",
    f"intra={torch.get_num_threads()}",
    f"interop={torch.get_num_interop_threads()}",
)

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = x.square().sum()
loss.backward()

print("loss:", loss.item())
print("gradient:", x.grad.tolist())
`,
  },
  training: {
    filename: "training.py",
    code: `import torch

torch.manual_seed(0)

x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = 2 * x + 1

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(30):
    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("loss:", round(loss.item(), 6))
print("weight:", round(model.weight.item(), 4))
print("bias:", round(model.bias.item(), 4))
`,
  },
  linalg: {
    filename: "linalg.py",
    code: `import torch

matrix = torch.tensor(
    [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
)

eigenvalues = torch.linalg.eigvalsh(matrix)
inverse = torch.linalg.inv(matrix)

print("matrix:")
print(matrix)
print("eigenvalues:", eigenvalues.tolist())
print("inverse check:")
print((matrix @ inverse).round(decimals=5))
`,
  },
});

const TORCH_COMPLETIONS = completeFromList([
  { label: "torch.tensor", type: "function", detail: "Create a tensor" },
  { label: "torch.arange", type: "function" },
  { label: "torch.zeros", type: "function" },
  { label: "torch.ones", type: "function" },
  { label: "torch.manual_seed", type: "function" },
  { label: "torch.nn", type: "module" },
  { label: "torch.nn.Linear", type: "class" },
  { label: "torch.nn.functional", type: "module" },
  { label: "torch.optim", type: "module" },
  { label: "torch.optim.SGD", type: "class" },
  { label: "torch.linalg", type: "module" },
  { label: "torch.linalg.inv", type: "function" },
  { label: "torch.linalg.eigvalsh", type: "function" },
]);

const STAGE_PROGRESS = {
  release: 4,
  pyodide: 15,
  dependencies: 35,
  filelock: 52,
  torch: 62,
  verify: 94,
  ready: 100,
};

const elements = {
  statusText: document.querySelector("#status-text"),
  statusDetail: document.querySelector("#status-detail"),
  statusDot: document.querySelector("#status-dot"),
  progressTrack: document.querySelector("#progress-track"),
  progressBar: document.querySelector("#progress-bar"),
  releaseValue: document.querySelector("#release-value"),
  torchValue: document.querySelector("#torch-value"),
  pyodideValue: document.querySelector("#pyodide-value"),
  wheelValue: document.querySelector("#wheel-value"),
  releaseLink: document.querySelector("#release-link"),
  output: document.querySelector("#output"),
  run: document.querySelector("#run-button"),
  stop: document.querySelector("#stop-button"),
  restart: document.querySelector("#restart-button"),
  copy: document.querySelector("#copy-button"),
  clear: document.querySelector("#clear-button"),
  resetCode: document.querySelector("#reset-code-button"),
  example: document.querySelector("#example-select"),
  filename: document.querySelector("#filename"),
  executionState: document.querySelector("#execution-state"),
};

let runtimeWorker = null;
let selectedRelease = null;
let runtimeReady = false;
let running = false;

function currentExample() {
  return EXAMPLES[elements.example.value] ?? EXAMPLES.autograd;
}

function replaceEditorText(value) {
  editor.dispatch({
    changes: { from: 0, to: editor.state.doc.length, insert: value },
    selection: { anchor: 0 },
    scrollIntoView: true,
  });
}

function runShortcut() {
  runPython();
  return true;
}

const editor = new EditorView({
  state: EditorState.create({
    doc: EXAMPLES.autograd.code,
    extensions: [
      basicSetup,
      python(),
      pythonLanguage.data.of({ autocomplete: TORCH_COMPLETIONS }),
      oneDark,
      Prec.high(
        keymap.of([
          indentWithTab,
          { key: "Mod-Enter", run: runShortcut },
          { key: "Shift-Enter", run: runShortcut },
        ]),
      ),
      EditorView.lineWrapping,
    ],
  }),
  parent: document.querySelector("#editor"),
});

function setStatus(label, detail, stage, state = "loading") {
  const progress = STAGE_PROGRESS[stage] ?? 4;
  elements.statusText.textContent = label;
  elements.statusDetail.textContent = detail;
  elements.progressBar.style.width = `${progress}%`;
  elements.progressTrack.setAttribute("aria-valuenow", String(progress));
  elements.statusDot.className = `status-dot is-${state}`;
}

function setExecutionState(value, state = "idle") {
  elements.executionState.textContent = value;
  elements.executionState.className = `execution-state${state === "idle" ? "" : ` is-${state}`}`;
}

function setControls() {
  elements.run.disabled = !runtimeReady || running;
  elements.stop.disabled = !running;
  elements.restart.disabled = !selectedRelease || running;
  elements.run.firstChild.textContent = running ? "Running " : "Run ";
}

function clearOutput() {
  elements.output.textContent = "";
}

function appendOutput(value, type = "stdout") {
  const span = document.createElement("span");
  span.className = `output-${type}`;
  span.textContent = String(value);
  elements.output.append(span);
  elements.output.scrollTop = elements.output.scrollHeight;
}

function formatBytes(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function validateReleaseManifest(manifest) {
  const configuration = manifest?.configuration;
  const wheel = manifest?.wheel;
  if (!configuration || !wheel) {
    throw new Error("The release manifest is missing configuration or wheel metadata.");
  }
  if (!configuration.release?.tag) {
    throw new Error("The release manifest does not identify its GitHub Release.");
  }
  if (!configuration.pyodide?.version || !configuration.pytorch?.version) {
    throw new Error("The release manifest is missing runtime version pins.");
  }
  if (!wheel.filename?.endsWith(".whl") || !wheel.sha256 || !wheel.size) {
    throw new Error("The release manifest contains invalid wheel metadata.");
  }
  if (wheel.filename.includes("/") || wheel.filename.includes("\\")) {
    throw new Error("The release manifest contains an unsafe wheel filename.");
  }
}

async function resolveLatestRelease() {
  const response = await fetch(PUBLISHED_MANIFEST_URL, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`The playground returned ${response.status} for its release manifest.`);
  }
  const manifest = await response.json();
  validateReleaseManifest(manifest);
  const releaseTag = manifest.configuration.release.tag;
  return {
    releaseTag,
    releaseUrl: `https://github.com/${REPOSITORY}/releases/tag/${encodeURIComponent(releaseTag)}`,
    wheelUrl: new URL(manifest.wheel.filename, RUNTIME_BASE_URL).href,
    wheelName: manifest.wheel.filename,
    wheelSize: manifest.wheel.size,
    wheelSha256: manifest.wheel.sha256,
    pyodideVersion: manifest.configuration.pyodide.version,
    torchVersion: manifest.configuration.pytorch.version,
  };
}

function showRelease(release) {
  elements.releaseValue.textContent = release.releaseTag;
  elements.releaseValue.title = release.releaseTag;
  elements.torchValue.textContent = release.torchVersion;
  elements.pyodideValue.textContent = release.pyodideVersion;
  elements.wheelValue.textContent = formatBytes(release.wheelSize);
  elements.wheelValue.title = `${release.wheelName}\nsha256: ${release.wheelSha256}`;
  elements.releaseLink.href = release.releaseUrl;
}

function handleWorkerMessage(event) {
  const message = event.data ?? {};
  switch (message.type) {
    case "status":
      setStatus(message.label, message.detail, message.stage);
      break;
    case "stdout":
      appendOutput(`${message.text}\n`);
      break;
    case "stderr":
      appendOutput(`${message.text}\n`, "error");
      break;
    case "ready":
      runtimeReady = true;
      running = false;
      setControls();
      setExecutionState("idle");
      setStatus(
        "Runtime ready",
        `torch ${message.details.version} · ${message.details.platform} · one thread`,
        "ready",
        "ready",
      );
      clearOutput();
      appendOutput(
        `Python runtime ready\ntorch ${message.details.version}\nPyodide ${selectedRelease.pyodideVersion}\n`,
        "meta",
      );
      break;
    case "run-started":
      running = true;
      setControls();
      setExecutionState("running", "running");
      break;
    case "result":
      appendOutput(`${message.text}\n`, "result");
      break;
    case "run-finished":
      running = false;
      setControls();
      setExecutionState("idle");
      break;
    case "fatal":
      runtimeReady = false;
      running = false;
      setControls();
      setExecutionState("failed", "error");
      setStatus("Runtime failed", "Use Restart runtime to try again.", "release", "error");
      appendOutput(`${message.error}\n`, "error");
      break;
  }
}

function startWorker(release) {
  runtimeWorker?.terminate();
  runtimeReady = false;
  running = false;
  setControls();
  setExecutionState("starting", "running");
  runtimeWorker = new Worker("./worker.js", { type: "module" });
  runtimeWorker.addEventListener("message", handleWorkerMessage);
  runtimeWorker.addEventListener("error", (event) => {
    handleWorkerMessage({ data: { type: "fatal", error: event.message || "Web Worker failed." } });
  });
  runtimeWorker.postMessage({ type: "init", config: release });
}

async function bootstrap() {
  clearOutput();
  appendOutput("Reading release manifest…\n", "meta");
  setStatus("Initializing runtime", "Reading the deployed release manifest.", "release");
  try {
    selectedRelease = await resolveLatestRelease();
  } catch (error) {
    selectedRelease = { ...FALLBACK_RELEASE };
    appendOutput(`Manifest lookup failed: ${error.message}\n`, "error");
    appendOutput(`Using pinned release ${selectedRelease.releaseTag}.\n`, "meta");
  }
  showRelease(selectedRelease);
  setControls();
  startWorker(selectedRelease);
}

function runPython() {
  if (!runtimeReady || running || !runtimeWorker) return;
  clearOutput();
  appendOutput(`$ python ${currentExample().filename}\n`, "meta");
  runtimeWorker.postMessage({ type: "run", code: editor.state.doc.toString() });
}

function restartRuntime() {
  if (!selectedRelease || running) return;
  clearOutput();
  appendOutput("Restarting Pyodide…\n", "meta");
  setStatus("Restarting runtime", "Creating a new worker.", "pyodide");
  startWorker(selectedRelease);
}

function stopExecution() {
  if (!running || !selectedRelease) return;
  runtimeWorker?.terminate();
  runtimeWorker = null;
  running = false;
  runtimeReady = false;
  clearOutput();
  appendOutput("Execution stopped. Restarting Pyodide…\n", "error");
  setStatus("Restarting runtime", "The previous worker was terminated.", "pyodide");
  startWorker(selectedRelease);
}

elements.run.addEventListener("click", runPython);
elements.stop.addEventListener("click", stopExecution);
elements.restart.addEventListener("click", restartRuntime);
elements.clear.addEventListener("click", clearOutput);
elements.resetCode.addEventListener("click", () => {
  replaceEditorText(currentExample().code);
  editor.focus();
});
elements.example.addEventListener("change", () => {
  const example = currentExample();
  elements.filename.textContent = example.filename;
  replaceEditorText(example.code);
  editor.focus();
});
elements.copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(editor.state.doc.toString());
  const previous = elements.copy.textContent;
  elements.copy.textContent = "Copied";
  window.setTimeout(() => {
    elements.copy.textContent = previous;
  }, 1200);
});

bootstrap();
