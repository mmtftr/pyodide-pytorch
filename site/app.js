import { acceptCompletion } from "@codemirror/autocomplete";
import { indentWithTab } from "@codemirror/commands";
import { python, pythonLanguage } from "@codemirror/lang-python";
import { EditorState, Prec } from "@codemirror/state";
import { oneDark } from "@codemirror/theme-one-dark";
import { hoverTooltip, keymap } from "@codemirror/view";
import { basicSetup, EditorView } from "codemirror";

const REPOSITORY = "mmtftr/pyodide-pytorch";
const CACHE_PREFIX = "pyodide-pytorch-playground-";
const ASSET_VERSION = "5";
const RUNTIME_BASE_URL = new URL("./runtime/", document.baseURI);
const PUBLISHED_MANIFEST_URL = new URL("build-manifest.json", RUNTIME_BASE_URL);

const FALLBACK_RELEASE = Object.freeze({
  releaseTag: "torch-2.13.0-pyodide-314.0.2-r4",
  releaseUrl:
    "https://github.com/mmtftr/pyodide-pytorch/releases/tag/torch-2.13.0-pyodide-314.0.2-r4",
  wheelUrl: new URL(
    "torch-2.13.0+pyodide314.0.2.r4-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
    RUNTIME_BASE_URL,
  ).href,
  wheelName:
    "torch-2.13.0+pyodide314.0.2.r4-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
  wheelSize: 27_205_688,
  wheelSha256: "f40cea64246a09d63d0ca10e5257bcb53ba1c9de442f674464b7289f01b292fa",
  pyodideVersion: "314.0.2",
  torchVersion: "2.13.0+pyodide314.0.2.r4",
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
  transformerBenchmark: {
    filename: "transformer_webgpu.py",
    code: `import time
import torch
import torch.nn.functional as F

if not torch.webgpu.is_available():
    raise RuntimeError("WebGPU is not available in this browser")

await torch.webgpu.init()

# A deterministic pre-norm decoder block. The shapes are intentionally small
# enough for an interactive browser check, but the data flow is real:
# embeddings -> QKV -> causal attention -> MLP -> logits.
BATCH, TOKENS, WIDTH, HEADS, FF_WIDTH, VOCAB = 1, 8, 16, 4, 32, 24
HEAD_DIM = WIDTH // HEADS
GPU_REPEATS = 5
CPU_REPEATS = 5

token_ids = torch.tensor([[1, 5, 2, 7, 3, 6, 4, 8]], dtype=torch.int32)
position_ids = torch.arange(TOKENS, dtype=torch.int32).reshape(1, -1)

def parameter(shape, scale, shift=0.0):
    count = 1
    for dimension in shape:
        count *= dimension
    values = (torch.arange(count, dtype=torch.float32) % 29) - 14
    return values.reshape(shape) * scale + shift

cpu_parameters = {
    "token": parameter((VOCAB, WIDTH), 0.015),
    "position": parameter((TOKENS, WIDTH), 0.01),
    "ln1_weight": parameter((WIDTH,), 0.01, 1.0),
    "ln1_bias": parameter((WIDTH,), 0.002),
    "qkv_weight": parameter((3 * WIDTH, WIDTH), 0.003),
    "qkv_bias": parameter((3 * WIDTH,), 0.001),
    "projection_weight": parameter((WIDTH, WIDTH), 0.004),
    "projection_bias": parameter((WIDTH,), 0.001),
    "ln2_weight": parameter((WIDTH,), 0.01, 1.0),
    "ln2_bias": parameter((WIDTH,), 0.002),
    "up_weight": parameter((FF_WIDTH, WIDTH), 0.003),
    "up_bias": parameter((FF_WIDTH,), 0.001),
    "down_weight": parameter((WIDTH, FF_WIDTH), 0.003),
    "down_bias": parameter((WIDTH,), 0.001),
    "final_weight": parameter((VOCAB, WIDTH), 0.004),
    "final_bias": parameter((VOCAB,), 0.001),
}

def decoder(ids, positions, weights):
    hidden = F.embedding(ids, weights["token"])
    hidden = hidden + F.embedding(positions, weights["position"])
    normalized = F.layer_norm(
        hidden, (WIDTH,), weights["ln1_weight"], weights["ln1_bias"], 1e-5
    )
    qkv = F.linear(normalized, weights["qkv_weight"], weights["qkv_bias"])
    qkv = qkv.view(BATCH, TOKENS, 3, HEADS, HEAD_DIM)
    query = qkv.select(2, 0).permute(0, 2, 1, 3)
    key = qkv.select(2, 1).permute(0, 2, 1, 3)
    value = qkv.select(2, 2).permute(0, 2, 1, 3)
    attended = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=True
    )
    attended = attended.permute(0, 2, 1, 3).contiguous()
    attended = attended.view(BATCH, TOKENS, WIDTH)
    hidden = hidden + F.linear(
        attended, weights["projection_weight"], weights["projection_bias"]
    )
    normalized = F.layer_norm(
        hidden, (WIDTH,), weights["ln2_weight"], weights["ln2_bias"], 1e-5
    )
    mlp = F.gelu(F.linear(normalized, weights["up_weight"], weights["up_bias"]))
    hidden = hidden + F.linear(
        mlp, weights["down_weight"], weights["down_bias"]
    )
    return F.linear(hidden, weights["final_weight"], weights["final_bias"])

with torch.no_grad():
    cpu_logits = decoder(token_ids, position_ids, cpu_parameters)
    gpu_parameters = {
        name: value.to("webgpu") for name, value in cpu_parameters.items()
    }
    gpu_token_ids = token_ids.to("webgpu")
    gpu_position_ids = position_ids.to("webgpu")

    diagnostics_before = torch.webgpu.diagnostics()
    fallbacks_before = torch.webgpu.cpu_fallbacks()

    # Warm-up compiles and caches pipelines. It is excluded from timed work.
    gpu_logits = decoder(gpu_token_ids, gpu_position_ids, gpu_parameters)
    await torch.webgpu.synchronize()

    started = time.perf_counter()
    for _ in range(GPU_REPEATS):
        gpu_logits = decoder(gpu_token_ids, gpu_position_ids, gpu_parameters)
    await torch.webgpu.synchronize()
    gpu_ms = (time.perf_counter() - started) * 1_000 / GPU_REPEATS

    started = time.perf_counter()
    for _ in range(CPU_REPEATS):
        cpu_logits = decoder(token_ids, position_ids, cpu_parameters)
    cpu_ms = (time.perf_counter() - started) * 1_000 / CPU_REPEATS

    readback_started = time.perf_counter()
    result = await torch.webgpu.to_cpu_async(gpu_logits)
    readback_ms = (time.perf_counter() - readback_started) * 1_000

torch.testing.assert_close(result, cpu_logits, rtol=2e-4, atol=2e-5)
diagnostics_after = torch.webgpu.diagnostics()
fallbacks_after = torch.webgpu.cpu_fallbacks()
assert fallbacks_after == fallbacks_before, "transformer used a CPU fallback"

dispatches = diagnostics_after["dispatches"] - diagnostics_before["dispatches"]
submissions = (
    diagnostics_after["command_submissions"]
    - diagnostics_before["command_submissions"]
)

print(f"profile: tiny-gpt-b{BATCH}-t{TOKENS}-c{WIDTH}-h{HEADS}")
print(f"correctness: CPU and WebGPU logits match · checksum={result.sum().item():.6f}")
print(f"WebGPU: {gpu_ms:.2f} ms/forward · {BATCH * TOKENS * 1_000 / gpu_ms:.1f} tokens/s")
print(f"CPU:    {cpu_ms:.2f} ms/forward")
print(f"GPU/CPU time ratio: {gpu_ms / cpu_ms:.2f}x")
print(f"final asynchronous readback: {readback_ms:.2f} ms")
print(f"dispatches: {dispatches} · submissions: {submissions} · new CPU fallbacks: 0")
print("Note: software WebGPU adapters are regression tools, not hardware benchmarks.")
`,
  },
  webgpuBenchmark: {
    filename: "webgpu_benchmark.py",
    code: `import time
import torch

if not torch.webgpu.is_available():
    raise RuntimeError("WebGPU is not available in this browser")

await torch.webgpu.init()

elements = 1_048_576
iterations = 10
left_cpu = torch.arange(elements, dtype=torch.float32)
right_cpu = torch.full((elements,), 0.5, dtype=torch.float32)

started = time.perf_counter()
for _ in range(iterations):
    expected = left_cpu * right_cpu + left_cpu
cpu_ms = (time.perf_counter() - started) * 1_000

upload_started = time.perf_counter()
left = left_cpu.to("webgpu")
right = right_cpu.to("webgpu")
upload_ms = (time.perf_counter() - upload_started) * 1_000

# Warm up pipeline creation before timing.
warmup = left * right + left
await torch.webgpu.synchronize()

first_kernel = torch.webgpu.kernel_submissions()
started = time.perf_counter()
for _ in range(iterations):
    result = left * right + left
await torch.webgpu.synchronize()
webgpu_ms = (time.perf_counter() - started) * 1_000
kernels = torch.webgpu.kernel_submissions() - first_kernel

readback_started = time.perf_counter()
result_cpu = await torch.webgpu.to_cpu_async(result)
readback_ms = (time.perf_counter() - readback_started) * 1_000
torch.testing.assert_close(result_cpu, expected)

print(f"elements: {elements:,} · iterations: {iterations}")
print(f"CPU:    {cpu_ms:.1f} ms total · {cpu_ms / iterations:.2f} ms/iteration")
print(
    f"WebGPU: {webgpu_ms:.1f} ms total · "
    f"{webgpu_ms / iterations:.2f} ms/iteration"
)
print(f"compute speedup (transfers excluded): {cpu_ms / webgpu_ms:.2f}x")
print(f"one-time upload: {upload_ms:.1f} ms · final readback: {readback_ms:.1f} ms")
print(
    f"submitted kernels: {kernels} · "
    f"implicit CPU fallbacks: {torch.webgpu.cpu_fallbacks()}"
)
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
  {
    label: "torch.webgpu",
    type: "module",
    detail: "Experimental browser WebGPU backend",
  },
  {
    label: "torch.webgpu.init",
    type: "function",
    detail: "await torch.webgpu.init()",
  },
  {
    label: "torch.webgpu.synchronize",
    type: "function",
    detail: "await torch.webgpu.synchronize()",
  },
  {
    label: "torch.webgpu.to_cpu_async",
    type: "function",
    detail: "await torch.webgpu.to_cpu_async(tensor)",
  },
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
