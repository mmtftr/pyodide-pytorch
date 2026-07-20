import { acceptCompletion } from "@codemirror/autocomplete";
import { indentWithTab } from "@codemirror/commands";
import { python, pythonLanguage } from "@codemirror/lang-python";
import { EditorState, Prec } from "@codemirror/state";
import { oneDark } from "@codemirror/theme-one-dark";
import { hoverTooltip, keymap } from "@codemirror/view";
import { basicSetup, EditorView } from "codemirror";

const REPOSITORY = "mmtftr/pyodide-pytorch";
const CACHE_PREFIX = "pyodide-pytorch-playground-";
const ASSET_VERSION = "2";
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

const COMMON_COMPLETIONS = [
  {
    label: "print",
    type: "function",
    detail: "print(*objects, sep=' ', end='\\n', file=None, flush=False)",
    info: "Print objects to a text stream.",
  },
  { label: "len", type: "function", detail: "len(object)", info: "Return the number of items." },
  { label: "range", type: "class", detail: "range(stop) or range(start, stop, step)" },
  { label: "enumerate", type: "class", detail: "enumerate(iterable, start=0)" },
  { label: "zip", type: "class", detail: "zip(*iterables, strict=False)" },
  { label: "list", type: "class" },
  { label: "dict", type: "class" },
  { label: "set", type: "class" },
  { label: "tuple", type: "class" },
  { label: "sum", type: "function" },
  { label: "min", type: "function" },
  { label: "max", type: "function" },
  { label: "torch", type: "module", detail: "PyTorch package" },
  { label: "torch.tensor", type: "function", detail: "torch.tensor(data, *, dtype=None, device=None, requires_grad=False)" },
  { label: "torch.arange", type: "function", detail: "torch.arange(start=0, end, step=1, *, dtype=None)" },
  { label: "torch.zeros", type: "function" },
  { label: "torch.ones", type: "function" },
  { label: "torch.randn", type: "function" },
  { label: "torch.manual_seed", type: "function" },
  { label: "torch.no_grad", type: "class" },
  { label: "torch.nn", type: "module" },
  { label: "torch.nn.Linear", type: "class" },
  { label: "torch.nn.functional", type: "module" },
  { label: "torch.optim", type: "module" },
  { label: "torch.optim.SGD", type: "class" },
  { label: "torch.linalg", type: "module" },
  { label: "torch.linalg.inv", type: "function" },
  { label: "torch.linalg.eigvalsh", type: "function" },
];

const STATIC_DOCUMENTATION = new Map(
  COMMON_COMPLETIONS.filter((item) => item.detail || item.info).map((item) => [
    item.label,
    {
      symbol: item.label,
      signature: item.detail ?? item.label,
      documentation: item.info ?? "",
      module: item.label.startsWith("torch") ? "torch" : "builtins",
      qualname: item.label,
    },
  ]),
);

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
  cacheValue: document.querySelector("#cache-value"),
  releaseLink: document.querySelector("#release-link"),
  output: document.querySelector("#output"),
  run: document.querySelector("#run-button"),
  stop: document.querySelector("#stop-button"),
  restart: document.querySelector("#restart-button"),
  clearCache: document.querySelector("#clear-cache-button"),
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
let requestSequence = 0;

const pendingWorkerRequests = new Map();
const inspectionCache = new Map();
const completionCache = new Map();

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

function clearWorkerRequests() {
  for (const { resolve, timeout } of pendingWorkerRequests.values()) {
    window.clearTimeout(timeout);
    resolve(null);
  }
  pendingWorkerRequests.clear();
  inspectionCache.clear();
  completionCache.clear();
}

function requestWorker(type, payload) {
  if (!runtimeReady || running || !runtimeWorker) return Promise.resolve(null);
  const id = ++requestSequence;
  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => {
      pendingWorkerRequests.delete(id);
      resolve(null);
    }, 5000);
    pendingWorkerRequests.set(id, { resolve, timeout });
    runtimeWorker.postMessage({ type, id, ...payload });
  });
}

async function inspectExpression(expression) {
  if (inspectionCache.has(expression)) return inspectionCache.get(expression);
  const result = await requestWorker("inspect", { expression });
  if (result) inspectionCache.set(expression, result);
  return result;
}

async function completeExpression(expression, prefix) {
  const key = `${expression}\u0000${prefix}`;
  if (completionCache.has(key)) return completionCache.get(key);
  const result = await requestWorker("complete", { expression, prefix });
  const members = Array.isArray(result) ? result : [];
  completionCache.set(key, members);
  return members;
}

async function completionSource(context) {
  const token = context.matchBefore(/[A-Za-z_][\w.]*/);
  if (!token) {
    return context.explicit ? { from: context.pos, options: COMMON_COMPLETIONS } : null;
  }
  if (!context.explicit && token.from === token.to) return null;

  const lastDot = token.text.lastIndexOf(".");
  if (lastDot < 0 || !runtimeReady || running) {
    return {
      from: token.from,
      options: COMMON_COMPLETIONS,
      validFor: /^[\w.]*$/,
    };
  }

  const expression = token.text.slice(0, lastDot);
  const prefix = token.text.slice(lastDot + 1);
  const members = await completeExpression(expression, prefix);
  return {
    from: token.from + lastDot + 1,
    options: members.map((member) => ({
      label: member.name,
      type: member.kind,
      detail: member.signature || member.module || "",
      info: member.documentation || undefined,
    })),
    validFor: /^\w*$/,
  };
}

function symbolAt(state, position) {
  const line = state.doc.lineAt(position);
  const offset = position - line.from;
  let start = offset;
  let end = offset;
  while (start > 0 && /[\w.]/.test(line.text[start - 1])) start -= 1;
  while (end < line.text.length && /[\w.]/.test(line.text[end])) end += 1;
  const symbol = line.text.slice(start, end).replace(/^\.+|\.+$/g, "");
  if (!/^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/.test(symbol)) return null;
  return { symbol, from: line.from + start, to: line.from + end };
}

function documentationUrl(result) {
  if (result.symbol.startsWith("torch")) {
    return `https://pytorch.org/docs/stable/generated/${encodeURIComponent(result.symbol)}.html`;
  }
  if (result.module === "builtins") {
    return `https://docs.python.org/3/library/functions.html#${encodeURIComponent(result.symbol)}`;
  }
  if (result.module && !result.module.startsWith("__")) {
    return `https://docs.python.org/3/library/${encodeURIComponent(result.module)}.html`;
  }
  return null;
}

function createDocumentationDOM(result) {
  const container = document.createElement("div");
  container.className = "cm-doc-tooltip";

  const symbol = document.createElement("strong");
  symbol.textContent = result.symbol;
  container.append(symbol);

  if (result.signature) {
    const signature = document.createElement("code");
    signature.textContent = result.signature;
    container.append(signature);
  }

  if (result.documentation) {
    const documentation = document.createElement("p");
    documentation.textContent = result.documentation;
    container.append(documentation);
  }

  const url = documentationUrl(result);
  if (url) {
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = result.symbol.startsWith("torch") ? "PyTorch documentation" : "Python documentation";
    container.append(link);
  }

  return container;
}

const documentationTooltip = hoverTooltip(
  async (view, position) => {
    const target = symbolAt(view.state, position);
    if (!target) return null;
    const fallback = STATIC_DOCUMENTATION.get(target.symbol);
    const result = runtimeReady && !running ? await inspectExpression(target.symbol) : fallback;
    if (!result && !fallback) return null;
    const documentation = result ?? fallback;
    return {
      pos: target.from,
      end: target.to,
      above: true,
      create: () => ({ dom: createDocumentationDOM(documentation) }),
    };
  },
  { hoverTime: 350 },
);

const editor = new EditorView({
  state: EditorState.create({
    doc: EXAMPLES.autograd.code,
    extensions: [
      basicSetup,
      python(),
      pythonLanguage.data.of({ autocomplete: completionSource }),
      documentationTooltip,
      oneDark,
      Prec.high(
        keymap.of([
          { key: "Tab", run: acceptCompletion },
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
    case "request-result": {
      const request = pendingWorkerRequests.get(message.id);
      if (request) {
        window.clearTimeout(request.timeout);
        pendingWorkerRequests.delete(message.id);
        request.resolve(message.result ?? null);
      }
      break;
    }
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
      inspectionCache.clear();
      completionCache.clear();
      setControls();
      setExecutionState("idle");
      break;
    case "fatal":
      runtimeReady = false;
      running = false;
      clearWorkerRequests();
      setControls();
      setExecutionState("failed", "error");
      setStatus("Runtime failed", "Use Restart runtime to try again.", "release", "error");
      appendOutput(`${message.error}\n`, "error");
      break;
  }
}

function startWorker(release) {
  runtimeWorker?.terminate();
  clearWorkerRequests();
  runtimeReady = false;
  running = false;
  setControls();
  setExecutionState("starting", "running");
  runtimeWorker = new Worker(`./worker.js?v=${ASSET_VERSION}`, { type: "module" });
  runtimeWorker.addEventListener("message", handleWorkerMessage);
  runtimeWorker.addEventListener("error", (event) => {
    handleWorkerMessage({ data: { type: "fatal", error: event.message || "Web Worker failed." } });
  });
  runtimeWorker.postMessage({ type: "init", config: release });
}

async function initializeAssetCache() {
  if (!("serviceWorker" in navigator) || !("caches" in window)) {
    elements.cacheValue.textContent = "unavailable";
    elements.clearCache.disabled = true;
    return;
  }
  try {
    await navigator.serviceWorker.register(`./service-worker.js?v=${ASSET_VERSION}`);
    await navigator.serviceWorker.ready;
    elements.cacheValue.textContent = "enabled";
    elements.clearCache.disabled = false;
  } catch (error) {
    console.warn("Playground cache initialization failed", error);
    elements.cacheValue.textContent = "unavailable";
    elements.clearCache.disabled = true;
  }
}

async function clearAssetCache() {
  if (!("caches" in window)) return;
  elements.clearCache.disabled = true;
  const keys = await caches.keys();
  await Promise.all(keys.filter((key) => key.startsWith(CACHE_PREFIX)).map((key) => caches.delete(key)));
  elements.cacheValue.textContent = "cleared";
  window.setTimeout(() => {
    elements.cacheValue.textContent = "enabled";
    elements.clearCache.disabled = false;
  }, 1200);
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
  running = true;
  setControls();
  setExecutionState("running", "running");
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
  clearWorkerRequests();
  clearOutput();
  appendOutput("Execution stopped. Restarting Pyodide…\n", "error");
  setStatus("Restarting runtime", "The previous worker was terminated.", "pyodide");
  startWorker(selectedRelease);
}

elements.run.addEventListener("click", runPython);
elements.stop.addEventListener("click", stopExecution);
elements.restart.addEventListener("click", restartRuntime);
elements.clearCache.addEventListener("click", clearAssetCache);
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

initializeAssetCache().finally(bootstrap);
