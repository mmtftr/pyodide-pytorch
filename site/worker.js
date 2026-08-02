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

async function sha256Hex(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (value) => value.toString(16).padStart(2, "0")).join("");
}

function normalizePackageName(name) {
  return name.toLowerCase().replace(/[-_.]+/g, "-");
}

function validateTransformersManifest(manifest) {
  if (
    manifest?.schema_version !== 1 ||
    manifest?.requirements !== "config/transformers-browser-requirements.txt" ||
    manifest?.model_only !== true ||
    manifest?.tokenizers_included !== false ||
    !Array.isArray(manifest?.packages)
  ) {
    throw new Error("The Transformers browser manifest is invalid.");
  }
  const packages = new Map();
  for (const entry of manifest.packages) {
    if (
      typeof entry?.name !== "string" ||
      typeof entry?.version !== "string" ||
      typeof entry?.filename !== "string" ||
      !/^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$/.test(entry.filename) ||
      typeof entry?.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(entry.sha256) ||
      !Number.isSafeInteger(entry?.size) ||
      entry.size <= 0
    ) {
      throw new Error("The Transformers browser manifest has an invalid package entry.");
    }
    const normalizedName = normalizePackageName(entry.name);
    if (normalizedName === "tokenizers" || packages.has(normalizedName)) {
      throw new Error(`The Transformers browser manifest rejects ${entry.name}.`);
    }
    packages.set(normalizedName, entry);
  }
  for (const required of ["filelock", "huggingface-hub", "transformers"]) {
    if (!packages.has(required)) {
      throw new Error(`The Transformers browser manifest omits ${required}.`);
    }
  }
  return packages;
}

async function loadVerifiedPackage(entry, manifestURL) {
  const packageURL = new URL(entry.filename, manifestURL);
  const response = await fetch(packageURL, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Could not load ${entry.name}: ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength !== entry.size) {
    throw new Error(
      `${entry.name} wheel size mismatch: expected ${entry.size}, received ${bytes.byteLength}.`,
    );
  }
  const sha256 = await sha256Hex(bytes);
  if (sha256 !== entry.sha256) {
    throw new Error(
      `${entry.name} wheel SHA-256 mismatch: expected ${entry.sha256}, received ${sha256}.`,
    );
  }
  await pyodide.loadPackage(packageURL.href);
}

async function initialize(config) {
  try {
    postStatus("pyodide", "Loading Pyodide", `Runtime ${config.pyodideVersion}.`);
    const indexURL = `https://cdn.jsdelivr.net/pyodide/v${config.pyodideVersion}/full/`;
    const { loadPyodide } = await import(`${indexURL}pyodide.mjs`);
    pyodide = await loadPyodide({ indexURL });

    pyodide.setStdout({ batched: (text) => self.postMessage({ type: "stdout", text }) });
    pyodide.setStderr({ batched: (text) => self.postMessage({ type: "stderr", text }) });

    postStatus(
      "dependencies",
      "Loading Python dependencies",
      "PyTorch and model-only Transformers dependencies from the pinned Pyodide lock.",
    );
    await pyodide.loadPackage([
      "numpy",
      "typing-extensions",
      "sympy",
      "networkx",
      "jinja2",
      "fsspec",
      "pyyaml",
      "regex",
      "requests",
      "safetensors",
      "tqdm",
    ]);

    postStatus(
      "transformers",
      "Preparing model-only Transformers support",
      "Disabling the optional GGUF loader when compiled tokenizers is unavailable.",
    );
    const transformersBootstrapURL = new URL(
      "./transformers_browser_bootstrap.py",
      self.location.href,
    );
    const transformersBootstrapResponse = await fetch(transformersBootstrapURL, {
      cache: "no-cache",
    });
    if (!transformersBootstrapResponse.ok) {
      throw new Error(
        `Could not load Transformers bootstrap: ${transformersBootstrapResponse.status}`,
      );
    }
    pyodide.FS.writeFile(
      "transformers_browser_bootstrap.py",
      await transformersBootstrapResponse.text(),
    );
    const transformersQ8URL = new URL(
      "./transformers_q8.py",
      self.location.href,
    );
    const transformersQ8Response = await fetch(transformersQ8URL, {
      cache: "no-cache",
    });
    if (!transformersQ8Response.ok) {
      throw new Error(
        `Could not load Transformers Q8 helper: ${transformersQ8Response.status}`,
      );
    }
    pyodide.FS.writeFile(
      "transformers_q8.py",
      await transformersQ8Response.text(),
    );
    const transformersGemma2URL = new URL(
      "./transformers_gemma2_webgpu.py",
      self.location.href,
    );
    const transformersGemma2Response = await fetch(transformersGemma2URL, {
      cache: "no-cache",
    });
    if (!transformersGemma2Response.ok) {
      throw new Error(
        `Could not load Transformers Gemma2 helper: ${transformersGemma2Response.status}`,
      );
    }
    pyodide.FS.writeFile(
      "transformers_gemma2_webgpu.py",
      await transformersGemma2Response.text(),
    );
    await pyodide.runPythonAsync(`
import os

from transformers_browser_bootstrap import disable_optional_gguf_without_tokenizers

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
disable_optional_gguf_without_tokenizers()
`);

    postStatus("torch", "Verifying PyTorch wheel", config.wheelName);
    const wheelResponse = await fetch(config.wheelUrl, { cache: "no-cache" });
    if (!wheelResponse.ok) {
      throw new Error(`Could not load PyTorch wheel: ${wheelResponse.status}`);
    }
    const wheelBytes = await wheelResponse.arrayBuffer();
    if (wheelBytes.byteLength !== config.wheelSize) {
      throw new Error(
        `PyTorch wheel size mismatch: expected ${config.wheelSize}, received ${wheelBytes.byteLength}.`,
      );
    }
    const wheelSha256 = await sha256Hex(wheelBytes);
    if (wheelSha256 !== config.wheelSha256) {
      throw new Error(
        `PyTorch wheel SHA-256 mismatch: expected ${config.wheelSha256}, received ${wheelSha256}.`,
      );
    }

    postStatus("torch", "Loading PyTorch wheel", config.wheelName);
    await pyodide.loadPackage(config.wheelUrl);

    postStatus(
      "transformers",
      "Loading model-only Transformers",
      "Verifying and loading the deployed pure-Python wheels.",
    );
    const transformersManifestURL = new URL(
      "./runtime/transformers/transformers-browser-manifest.json",
      self.location.href,
    );
    const transformersManifestResponse = await fetch(transformersManifestURL, {
      cache: "no-cache",
    });
    if (!transformersManifestResponse.ok) {
      throw new Error(
        `Could not load Transformers browser manifest: ${transformersManifestResponse.status}`,
      );
    }
    const transformersPackages = validateTransformersManifest(
      await transformersManifestResponse.json(),
    );
    for (const entry of transformersPackages.values()) {
      await loadVerifiedPackage(entry, transformersManifestURL);
    }

    const webgpuAdapter = await navigator.gpu?.requestAdapter({
      powerPreference: "high-performance",
    });
    const webgpuAdapterInfo = webgpuAdapter?.info ?? {};
    self.__torchWebGPUFixed32Subgroups = Boolean(
      webgpuAdapter?.features.has("subgroups") &&
        webgpuAdapterInfo.subgroupMinSize === 32 &&
        webgpuAdapterInfo.subgroupMaxSize === 32,
    );

    const transformersFixtureURL = new URL(
      "./runtime/transformers/transformers_tiny.json",
      self.location.href,
    );
    const transformersFixtureResponse = await fetch(transformersFixtureURL, {
      cache: "no-cache",
    });
    if (!transformersFixtureResponse.ok) {
      throw new Error(
        `Could not load Transformers fixture: ${transformersFixtureResponse.status}`,
      );
    }
    const transformersFixtureSource = await transformersFixtureResponse.text();
    const transformersFixture = JSON.parse(transformersFixtureSource);
    const expectedTransformers = transformersPackages.get("transformers");
    if (transformersFixture.transformers_version !== expectedTransformers.version) {
      throw new Error(
        "The Transformers fixture version does not match the deployed wheel manifest.",
      );
    }
    pyodide.FS.writeFile("transformers_tiny.json", transformersFixtureSource);

    postStatus(
      "verify",
      "Verifying runtime",
      "Checking PyTorch, Transformers, platform, and thread invariants.",
    );
    const detailsJson = await pyodide.runPythonAsync(`
import importlib.metadata
import importlib.util
import json
import sys

import huggingface_hub
import js
import torch
import transformers
from transformers.utils import is_tokenizers_available
from transformers_browser_bootstrap import (
    disable_optional_gguf_without_tokenizers,
    enable_webgpu_opt_sdpa_mask_compatibility,
    enable_webgpu_preallocated_kv_cache,
    enable_webgpu_rms_norm_fusion,
    enable_webgpu_rotary_scaling_compatibility,
    enable_webgpu_swiglu_fusion,
)
from transformers_q8 import (
    Q8_FORMAT_VERSION,
    Q8_GROUP_SIZE,
    convert_linear_modules_q8_,
)
from transformers_gemma2_webgpu import (
    enable_webgpu_gemma2_rms_norm,
    enable_webgpu_gemma2_scalar_normalizer,
)

webgpu_rms_norm_fusion = enable_webgpu_rms_norm_fusion()
webgpu_rotary_scaling_compatibility = (
    enable_webgpu_rotary_scaling_compatibility()
)
webgpu_swiglu_fusion = enable_webgpu_swiglu_fusion()
webgpu_opt_sdpa_mask_compatibility = (
    enable_webgpu_opt_sdpa_mask_compatibility()
)
webgpu_preallocated_kv_cache = enable_webgpu_preallocated_kv_cache()
webgpu_gemma2_rms_norm = enable_webgpu_gemma2_rms_norm()
webgpu_gemma2_scalar_normalizer = (
    enable_webgpu_gemma2_scalar_normalizer()
)
webgpu_q8_device_supported = bool(js.globalThis.__torchWebGPUFixed32Subgroups)
webgpu_q8_linear = {
    "enabled": webgpu_q8_device_supported,
    "operator_available": callable(torch.ops.webgpu.q8_linear),
    "helper_available": callable(convert_linear_modules_q8_),
    "device_supported": webgpu_q8_device_supported,
    "profile": "webgpu-q8-group128-v1-decode",
    "group_size": Q8_GROUP_SIZE,
    "format_version": Q8_FORMAT_VERSION,
    "decode_only": True,
}

json.dumps({
    "version": torch.__version__,
    "platform": sys.platform,
    "intra_threads": torch.get_num_threads(),
    "interop_threads": torch.get_num_interop_threads(),
    "transformers_version": transformers.__version__,
    "huggingface_hub_version": importlib.metadata.version("huggingface-hub"),
    "tokenizers_available": (
        importlib.util.find_spec("tokenizers") is not None
        or is_tokenizers_available()
    ),
    "model_only_bootstrap": disable_optional_gguf_without_tokenizers(),
    "webgpu_rms_norm_fusion": webgpu_rms_norm_fusion,
    "webgpu_rotary_scaling_compatibility": (
        webgpu_rotary_scaling_compatibility
    ),
    "webgpu_swiglu_fusion": webgpu_swiglu_fusion,
    "webgpu_opt_sdpa_mask_compatibility": (
        webgpu_opt_sdpa_mask_compatibility
    ),
    "webgpu_preallocated_kv_cache": webgpu_preallocated_kv_cache,
    "webgpu_gemma2_rms_norm": webgpu_gemma2_rms_norm,
    "webgpu_gemma2_scalar_normalizer": webgpu_gemma2_scalar_normalizer,
    "webgpu_q8_linear": webgpu_q8_linear,
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
    if (
      details.transformers_version !== transformersPackages.get("transformers").version ||
      details.huggingface_hub_version !==
        transformersPackages.get("huggingface-hub").version
    ) {
      throw new Error("The loaded Transformers package versions do not match the manifest.");
    }
    if (details.tokenizers_available || !details.model_only_bootstrap) {
      throw new Error("The runtime violated its model-only Transformers invariant.");
    }
    const rmsNormFusion = details.webgpu_rms_norm_fusion;
    if (
      rmsNormFusion?.enabled !== true ||
      rmsNormFusion.profile !== "transformers-4.46.3-webgpu-rms-norm" ||
      rmsNormFusion.transformers_version !== details.transformers_version ||
      rmsNormFusion.newly_patched + rmsNormFusion.already_patched !== 4 ||
      rmsNormFusion.targets?.length !== 4
    ) {
      throw new Error("The pinned Transformers WebGPU RMSNorm adapter was not enabled.");
    }
    const rotaryScaling = details.webgpu_rotary_scaling_compatibility;
    if (
      rotaryScaling?.enabled !== true ||
      rotaryScaling.profile !== "transformers-4.46.3-webgpu-rotary-scaling" ||
      rotaryScaling.transformers_version !== details.transformers_version ||
      rotaryScaling.newly_patched + rotaryScaling.already_patched !== 2 ||
      rotaryScaling.targets?.length !== 2 ||
      rotaryScaling.identity_dispatches_saved_per_call !== 2
    ) {
      throw new Error("The pinned Transformers WebGPU rotary adapter was not enabled.");
    }
    const swigluFusion = details.webgpu_swiglu_fusion;
    if (
      swigluFusion?.enabled !== true ||
      swigluFusion.profile !== "transformers-4.46.3-webgpu-decode-swiglu" ||
      swigluFusion.transformers_version !== details.transformers_version ||
      swigluFusion.newly_patched + swigluFusion.already_patched !== 3 ||
      swigluFusion.targets?.length !== 3 ||
      swigluFusion.upstream_dispatches_before_down_proj !== 4 ||
      swigluFusion.fused_dispatches_before_down_proj !== 1 ||
      swigluFusion.dispatches_saved_per_fused_call !== 3
    ) {
      throw new Error("The pinned Transformers WebGPU SwiGLU adapter was not enabled.");
    }
    const optSdpaMask = details.webgpu_opt_sdpa_mask_compatibility;
    if (
      optSdpaMask?.enabled !== true ||
      optSdpaMask.profile !== "transformers-4.46.3-webgpu-opt-sdpa-mask" ||
      optSdpaMask.transformers_version !== details.transformers_version ||
      optSdpaMask.newly_patched + optSdpaMask.already_patched !== 1 ||
      optSdpaMask.targets?.length !== 1 ||
      optSdpaMask.host_mask_truth_readbacks_avoided_per_eligible_call !== 1
    ) {
      throw new Error("The pinned Transformers WebGPU OPT SDPA-mask adapter was not enabled.");
    }
    const preallocatedKv = details.webgpu_preallocated_kv_cache;
    if (
      preallocatedKv?.enabled !== true ||
      preallocatedKv.profile !== "transformers-4.46.3-webgpu-preallocated-kv" ||
      preallocatedKv.transformers_version !== details.transformers_version ||
      preallocatedKv.newly_registered + preallocatedKv.already_registered !== 1 ||
      preallocatedKv.cache_implementation !== "webgpu_preallocated" ||
      preallocatedKv.dispatches_saved_per_layer_per_token !== 3
    ) {
      throw new Error("The pinned Transformers WebGPU preallocated KV cache was not registered.");
    }
    const q8Linear = details.webgpu_q8_linear;
    const gemma2RmsNorm = details.webgpu_gemma2_rms_norm;
    if (
      gemma2RmsNorm?.enabled !== true ||
      gemma2RmsNorm.profile !== "transformers-4.46.3-webgpu-gemma2-rms-norm" ||
      gemma2RmsNorm.transformers_version !== details.transformers_version ||
      gemma2RmsNorm.newly_patched + gemma2RmsNorm.already_patched !== 1 ||
      gemma2RmsNorm.dispatches_saved_per_call !== 5
    ) {
      throw new Error("The pinned Transformers Gemma2 RMSNorm adapter was not enabled.");
    }
    const gemma2Scalar = details.webgpu_gemma2_scalar_normalizer;
    if (
      gemma2Scalar?.enabled !== true ||
      gemma2Scalar.profile !==
        "transformers-4.46.3-webgpu-gemma2-scalar-normalizer" ||
      gemma2Scalar.transformers_version !== details.transformers_version ||
      gemma2Scalar.newly_patched + gemma2Scalar.already_patched !== 1 ||
      gemma2Scalar.mixed_device_mul_avoided_per_forward !== 1
    ) {
      throw new Error("The pinned Transformers Gemma2 scalar adapter was not enabled.");
    }
    if (
      q8Linear?.operator_available !== true ||
      q8Linear.helper_available !== true ||
      q8Linear.enabled !== q8Linear.device_supported ||
      q8Linear.profile !== "webgpu-q8-group128-v1-decode" ||
      q8Linear.group_size !== 128 ||
      q8Linear.format_version !== 1 ||
      q8Linear.decode_only !== true
    ) {
      throw new Error("The opt-in WebGPU Q8 linear helper was not loaded.");
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
