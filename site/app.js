const REPOSITORY = "mmtftr/pyodide-pytorch";
const LATEST_RELEASE_API = `https://api.github.com/repos/${REPOSITORY}/releases/latest`;

const FALLBACK_RELEASE = Object.freeze({
  releaseTag: "torch-2.13.0-pyodide-314.0.2-r1",
  releaseUrl:
    "https://github.com/mmtftr/pyodide-pytorch/releases/tag/torch-2.13.0-pyodide-314.0.2-r1",
  manifestUrl:
    "https://github.com/mmtftr/pyodide-pytorch/releases/download/torch-2.13.0-pyodide-314.0.2-r1/build-manifest.json",
  wheelUrl:
    "https://github.com/mmtftr/pyodide-pytorch/releases/download/torch-2.13.0-pyodide-314.0.2-r1/torch-2.13.0%2Bpyodide314.0.2-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
  wheelName:
    "torch-2.13.0+pyodide314.0.2-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
  wheelSize: 25_038_799,
  wheelSha256: "8691f0276528a7deee66c3abae3d21824ff8f2d20c9173142957bf04334af2a3",
  pyodideVersion: "314.0.2",
  torchVersion: "2.13.0+pyodide314.0.2",
  torchRef: "cf30153c4c131c8164ee7798e5022d810682e2cb",
});

const STAGE_PROGRESS = {
  release: 4,
  pyodide: 14,
  dependencies: 34,
  filelock: 52,
  torch: 60,
  verify: 92,
  ready: 100,
};

const elements = {
  statusText: document.querySelector("#status-text"),
  statusDetail: document.querySelector("#status-detail"),
  statusDot: document.querySelector("#status-dot"),
  progressTrack: document.querySelector("#progress-track"),
  progressBar: document.querySelector("#progress-bar"),
  releaseValue: document.querySelector("#release-value"),
  pyodideValue: document.querySelector("#pyodide-value"),
  wheelValue: document.querySelector("#wheel-value"),
  releaseLink: document.querySelector("#release-link"),
  code: document.querySelector("#python-code"),
  output: document.querySelector("#output"),
  run: document.querySelector("#run-button"),
  reset: document.querySelector("#reset-button"),
  copy: document.querySelector("#copy-button"),
  clear: document.querySelector("#clear-button"),
};

let runtimeWorker = null;
let selectedRelease = null;
let runtimeReady = false;
let running = false;

function setStatus(label, detail, stage, state = "loading") {
  const progress = STAGE_PROGRESS[stage] ?? 4;
  elements.statusText.textContent = label;
  elements.statusDetail.textContent = detail;
  elements.progressBar.style.width = `${progress}%`;
  elements.progressTrack.setAttribute("aria-valuenow", String(progress));
  elements.statusDot.className = `status-dot is-${state}`;
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

function validateReleaseManifest(release, manifest) {
  const configuration = manifest?.configuration;
  const wheel = manifest?.wheel;
  if (!configuration || !wheel) {
    throw new Error("The release manifest is missing configuration or wheel metadata.");
  }
  if (configuration.release?.tag !== release.tag_name) {
    throw new Error("The latest release tag does not match its build manifest.");
  }
  if (!configuration.pyodide?.version || !configuration.pytorch?.version) {
    throw new Error("The release manifest is missing runtime version pins.");
  }
  if (!wheel.filename?.endsWith(".whl") || !wheel.sha256 || !wheel.size) {
    throw new Error("The release manifest contains invalid wheel metadata.");
  }
}

async function resolveLatestRelease() {
  const response = await fetch(LATEST_RELEASE_API, {
    headers: { Accept: "application/vnd.github+json" },
  });
  if (!response.ok) {
    throw new Error(`GitHub returned ${response.status} while resolving the latest release.`);
  }

  const release = await response.json();
  const manifestAsset = release.assets?.find((asset) => asset.name === "build-manifest.json");
  if (!manifestAsset) {
    throw new Error("The latest release does not contain build-manifest.json.");
  }

  const manifestResponse = await fetch(manifestAsset.browser_download_url);
  if (!manifestResponse.ok) {
    throw new Error(`GitHub returned ${manifestResponse.status} for the release manifest.`);
  }
  const manifest = await manifestResponse.json();
  validateReleaseManifest(release, manifest);

  const wheelAsset = release.assets.find((asset) => asset.name === manifest.wheel.filename);
  if (!wheelAsset) {
    throw new Error("The wheel named by the release manifest is not attached to the release.");
  }

  return {
    releaseTag: release.tag_name,
    releaseUrl: release.html_url,
    manifestUrl: manifestAsset.browser_download_url,
    wheelUrl: wheelAsset.browser_download_url,
    wheelName: wheelAsset.name,
    wheelSize: wheelAsset.size,
    wheelSha256: manifest.wheel.sha256,
    pyodideVersion: manifest.configuration.pyodide.version,
    torchVersion: manifest.configuration.pytorch.version,
    torchRef: manifest.configuration.pytorch.ref,
  };
}

function showRelease(release) {
  elements.releaseValue.textContent = release.releaseTag;
  elements.releaseValue.title = release.releaseTag;
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
      elements.run.disabled = false;
      elements.reset.disabled = false;
      setStatus(
        "Ready to run Python",
        `torch ${message.details.version} · ${message.details.platform} · one worker thread`,
        "ready",
        "ready",
      );
      clearOutput();
      appendOutput(
        `Ready · torch ${message.details.version} · Pyodide ${selectedRelease.pyodideVersion}\n`,
        "meta",
      );
      appendOutput(`Wheel verified by release manifest: ${selectedRelease.wheelSha256}\n`, "meta");
      break;
    case "run-started":
      running = true;
      elements.run.disabled = true;
      elements.run.textContent = "Running…";
      break;
    case "result":
      appendOutput(`${message.text}\n`, "result");
      break;
    case "run-finished":
      running = false;
      elements.run.disabled = !runtimeReady;
      elements.run.innerHTML = '<span aria-hidden="true">▶</span> Run';
      break;
    case "fatal":
      runtimeReady = false;
      running = false;
      elements.run.disabled = true;
      elements.reset.disabled = false;
      setStatus("Runtime failed", "Restart the runtime to try again.", "release", "error");
      appendOutput(`${message.error}\n`, "error");
      break;
  }
}

function startWorker(release) {
  runtimeWorker?.terminate();
  runtimeReady = false;
  running = false;
  elements.run.disabled = true;
  elements.reset.disabled = false;
  runtimeWorker = new Worker("./worker.js", { type: "module" });
  runtimeWorker.addEventListener("message", handleWorkerMessage);
  runtimeWorker.addEventListener("error", (event) => {
    handleWorkerMessage({ data: { type: "fatal", error: event.message || "Web Worker failed." } });
  });
  runtimeWorker.postMessage({ type: "init", config: release });
}

async function bootstrap() {
  clearOutput();
  appendOutput("Resolving the latest verified GitHub Release…\n", "meta");
  setStatus(
    "Resolving latest release",
    "Reading wheel and ABI metadata from GitHub.",
    "release",
  );

  try {
    selectedRelease = await resolveLatestRelease();
    appendOutput(`Selected ${selectedRelease.releaseTag}.\n`, "meta");
  } catch (error) {
    selectedRelease = { ...FALLBACK_RELEASE };
    appendOutput(
      `Latest-release lookup failed; using verified fallback ${selectedRelease.releaseTag}.\n`,
      "error",
    );
    appendOutput(`${error.message}\n`, "meta");
  }

  showRelease(selectedRelease);
  startWorker(selectedRelease);
}

function runPython() {
  if (!runtimeReady || running || !runtimeWorker) return;
  clearOutput();
  appendOutput("# running playground.py\n", "meta");
  runtimeWorker.postMessage({ type: "run", code: elements.code.value });
}

elements.run.addEventListener("click", runPython);
elements.reset.addEventListener("click", () => {
  if (!selectedRelease) return;
  clearOutput();
  appendOutput("Restarting the WebAssembly runtime…\n", "meta");
  setStatus("Restarting runtime", "Creating a fresh Pyodide worker.", "pyodide");
  startWorker(selectedRelease);
});
elements.clear.addEventListener("click", clearOutput);
elements.copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(elements.code.value);
  const previous = elements.copy.textContent;
  elements.copy.textContent = "Copied";
  window.setTimeout(() => {
    elements.copy.textContent = previous;
  }, 1200);
});
elements.code.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    runPython();
  }
});

bootstrap();
