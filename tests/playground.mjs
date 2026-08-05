import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import puppeteer from "puppeteer-core";

const [publicArgument, expectedVersion, expectedReleaseTag, modeArgument] =
  process.argv.slice(2);
if (!publicArgument || !expectedVersion || !expectedReleaseTag) {
  throw new Error(
    "usage: node tests/playground.mjs PUBLIC_DIR EXPECTED_VERSION EXPECTED_RELEASE_TAG [--validate-only]",
  );
}
if (modeArgument && modeArgument !== "--validate-only") {
  throw new Error(`unsupported playground test mode: ${modeArgument}`);
}
const validateOnly = modeArgument === "--validate-only";
const adapterMode = process.env.WEBGPU_ADAPTER ?? "swiftshader";
if (!new Set(["hardware", "swiftshader"]).has(adapterMode)) {
  throw new Error(`unsupported WEBGPU_ADAPTER mode: ${adapterMode}`);
}

const publicDirectory = path.resolve(publicArgument);
if (!fs.existsSync(publicDirectory)) {
  throw new Error(`public directory does not exist: ${publicDirectory}`);
}
const sha256File = (filename) =>
  crypto.createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
const normalizePackageName = (name) =>
  name.toLowerCase().replace(/[-_.]+/g, "-");
const runtimeDirectory = path.join(publicDirectory, "runtime");
const releaseManifestPath = path.join(runtimeDirectory, "build-manifest.json");
if (!fs.existsSync(releaseManifestPath)) {
  throw new Error("release build manifest was not deployed");
}
const releaseManifest = JSON.parse(
  fs.readFileSync(releaseManifestPath, "utf8"),
);
const wheelRecord = releaseManifest.wheel;
if (
  releaseManifest.schema_version !== 1 ||
  releaseManifest.configuration?.release?.tag !== expectedReleaseTag ||
  releaseManifest.configuration?.pytorch?.version !== expectedVersion ||
  typeof wheelRecord?.filename !== "string" ||
  !/^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$/.test(wheelRecord.filename) ||
  !/^[0-9a-f]{64}$/.test(wheelRecord.sha256) ||
  !Number.isSafeInteger(wheelRecord.size) ||
  wheelRecord.size <= 0
) {
  throw new Error("release build manifest is invalid or does not match expectations");
}
const wheelPath = path.join(runtimeDirectory, wheelRecord.filename);
if (
  !fs.existsSync(wheelPath) ||
  fs.statSync(wheelPath).size !== wheelRecord.size ||
  sha256File(wheelPath) !== wheelRecord.sha256
) {
  throw new Error("deployed PyTorch wheel does not match its build manifest");
}
const checksumPath = `${wheelPath}.sha256`;
if (
  !fs.existsSync(checksumPath) ||
  fs.readFileSync(checksumPath, "utf8") !==
    `${wheelRecord.sha256}  ${wheelRecord.filename}\n`
) {
  throw new Error("deployed PyTorch checksum companion is invalid");
}
const transformersManifestPath = path.join(
  publicDirectory,
  "runtime",
  "transformers",
  "transformers-browser-manifest.json",
);
if (!fs.existsSync(transformersManifestPath)) {
  throw new Error("Transformers browser manifest was not deployed");
}
const transformersManifest = JSON.parse(
  fs.readFileSync(transformersManifestPath, "utf8"),
);
if (
  transformersManifest.schema_version !== 1 ||
  transformersManifest.requirements !==
    "config/transformers-browser-requirements.txt" ||
  transformersManifest.model_only !== true ||
  transformersManifest.tokenizers_included !== false ||
  !Array.isArray(transformersManifest.packages)
) {
  throw new Error("Transformers browser manifest is invalid");
}
const transformerNames = new Set();
for (const entry of transformersManifest.packages) {
  const normalizedName = normalizePackageName(entry?.name ?? "");
  const filename = entry?.filename;
  if (
    !normalizedName ||
    normalizedName === "tokenizers" ||
    transformerNames.has(normalizedName) ||
    typeof filename !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$/.test(filename) ||
    !/^[0-9a-f]{64}$/.test(entry.sha256) ||
    !Number.isSafeInteger(entry.size) ||
    entry.size <= 0
  ) {
    throw new Error("Transformers browser manifest has an invalid package entry");
  }
  const filenamePath = path.join(path.dirname(transformersManifestPath), filename);
  if (
    !fs.existsSync(filenamePath) ||
    fs.statSync(filenamePath).size !== entry.size ||
    sha256File(filenamePath) !== entry.sha256
  ) {
    throw new Error(`${entry.name} does not match the Transformers manifest`);
  }
  transformerNames.add(normalizedName);
}
for (const required of ["filelock", "huggingface-hub", "transformers"]) {
  if (!transformerNames.has(required)) {
    throw new Error(`Transformers browser manifest omits ${required}`);
  }
}
const transformersPackage = transformersManifest.packages?.find(
  (entry) => normalizePackageName(entry.name) === "transformers",
);
if (!transformersPackage) {
  throw new Error("Transformers browser manifest is not model-only");
}
const transformersFixturePath = path.join(
  path.dirname(transformersManifestPath),
  "transformers_tiny.json",
);
const transformersFixture = JSON.parse(
  fs.readFileSync(transformersFixturePath, "utf8"),
);
if (transformersFixture.transformers_version !== transformersPackage.version) {
  throw new Error("Transformers fixture version does not match its wheel manifest");
}
if (validateOnly) {
  console.log(
    JSON.stringify(
      {
        playground: "validated",
        release: expectedReleaseTag,
        torch: expectedVersion,
        wheel: wheelRecord.filename,
        transformers: transformersPackage.version,
      },
      null,
      2,
    ),
  );
  process.exit(0);
}

const chromeCandidates = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
];
const chrome = chromeCandidates.find(
  (candidate) => candidate && fs.existsSync(candidate),
);
if (!chrome) throw new Error("Chrome executable was not found");

const contentTypes = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json"],
  [".map", "application/json"],
  [".whl", "application/zip"],
]);
const requests = new Map();
const server = http.createServer((request, response) => {
  const url = new URL(request.url, "http://127.0.0.1");
  const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
  if (pathname === "/favicon.ico") {
    response.writeHead(204);
    response.end();
    return;
  }
  const filename = path.resolve(publicDirectory, `.${pathname}`);
  requests.set(pathname, (requests.get(pathname) ?? 0) + 1);
  if (
    !filename.startsWith(`${publicDirectory}${path.sep}`) ||
    !fs.existsSync(filename) ||
    !fs.statSync(filename).isFile()
  ) {
    response.writeHead(404);
    response.end("not found");
    return;
  }
  response.setHeader(
    "Content-Type",
    contentTypes.get(path.extname(filename)) ?? "application/octet-stream",
  );
  fs.createReadStream(filename).pipe(response);
});

let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const browserArguments = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
    "--disable-dawn-features=use_dxc",
    "--enable-webgpu-developer-features",
    "--use-gpu-in-tests",
    "--enable-accelerated-2d-canvas",
  ];
  if (adapterMode === "swiftshader") {
    browserArguments.push(
      "--enable-unsafe-swiftshader",
      "--use-webgpu-adapter=swiftshader",
    );
  }
  browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args: browserArguments,
  });
  const page = await browser.newPage();
  const browserErrors = [];
  const huggingFaceRequests = [];
  const pythonPackageIndexRequests = [];
  page.on("pageerror", (error) => browserErrors.push(String(error)));
  page.on("request", (request) => {
    const hostname = new URL(request.url()).hostname;
    if (
      hostname === "huggingface.co" ||
      hostname.endsWith(".huggingface.co") ||
      hostname === "hf.co" ||
      hostname.endsWith(".hf.co")
    ) {
      huggingFaceRequests.push(request.url());
    }
    if (
      hostname === "pypi.org" ||
      hostname.endsWith(".pypi.org") ||
      hostname === "pythonhosted.org" ||
      hostname.endsWith(".pythonhosted.org")
    ) {
      pythonPackageIndexRequests.push(request.url());
    }
  });
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  await page.goto(`http://127.0.0.1:${address.port}/`);
  await page.waitForFunction(
    () =>
      ["Runtime ready", "Runtime failed", "Release unavailable"].includes(
        document.querySelector("#status-text")?.textContent,
      ),
    { timeout: 300_000 },
  );
  const runtime = await page.evaluate(() => ({
    status: document.querySelector("#status-text")?.textContent,
    detail: document.querySelector("#status-detail")?.textContent,
    release: document.querySelector("#release-value")?.textContent,
    torch: document.querySelector("#torch-value")?.textContent,
    output: document.querySelector("#output")?.textContent,
    runEnabled: !document.querySelector("#run-button")?.disabled,
  }));
  if (runtime.status !== "Runtime ready") {
    throw new Error(
      `playground initialization failed: ${JSON.stringify(runtime)}`,
    );
  }
  if (runtime.release !== expectedReleaseTag) {
    throw new Error(`expected release ${expectedReleaseTag}, got ${runtime.release}`);
  }
  if (runtime.torch !== expectedVersion) {
    throw new Error(`expected torch ${expectedVersion}, got ${runtime.torch}`);
  }
  if (!runtime.runEnabled) throw new Error("Run button remained disabled");
  if (!runtime.output.includes(`Transformers ${transformersPackage.version}`)) {
    throw new Error(`runtime omitted pinned Transformers ${transformersPackage.version}`);
  }
  if (
    !runtime.output.includes(
      "WebGPU RMSNorm adapter transformers-4.46.3-webgpu-rms-norm",
    )
  ) {
    throw new Error("runtime omitted the pinned WebGPU RMSNorm adapter diagnostic");
  }
  if (
    !runtime.output.includes(
      "WebGPU rotary adapter transformers-4.46.3-webgpu-rotary-scaling",
    )
  ) {
    throw new Error("runtime omitted the pinned WebGPU rotary adapter diagnostic");
  }
  if (
    !runtime.output.includes(
      "WebGPU decode SwiGLU adapter " +
        "transformers-4.46.3-webgpu-decode-swiglu",
    )
  ) {
    throw new Error("runtime omitted the pinned WebGPU SwiGLU adapter diagnostic");
  }
  if (
    !runtime.output.includes(
      "WebGPU OPT SDPA-mask adapter " +
        "transformers-4.46.3-webgpu-opt-sdpa-mask",
    )
  ) {
    throw new Error("runtime omitted the pinned WebGPU OPT adapter diagnostic");
  }
  if (
    !runtime.output.includes(
      "WebGPU preallocated KV cache " +
        "transformers-4.46.3-webgpu-preallocated-kv",
    )
  ) {
    throw new Error("runtime omitted the pinned WebGPU preallocated KV diagnostic");
  }
  if (
    !runtime.output.includes(
      "WebGPU Q8 linear webgpu-q8-group128-v1-decode",
    )
  ) {
    throw new Error("runtime omitted the WebGPU Q8 linear diagnostic");
  }
  if (
    !runtime.output.includes(
      "WebGPU Gemma2 RMSNorm " +
        "transformers-4.46.3-webgpu-gemma2-rms-norm",
    ) ||
    !runtime.output.includes(
      "WebGPU Gemma2 scalar adapter " +
        "transformers-4.46.3-webgpu-gemma2-scalar-normalizer",
    )
  ) {
    throw new Error("runtime omitted the pinned WebGPU Gemma2 diagnostics");
  }

  const waitForExampleOutput = async (marker, timeout) => {
    try {
      await page.waitForFunction(
        (expected) =>
          document.querySelector("#output")?.textContent.includes(expected),
        { timeout },
        marker,
      );
    } catch (error) {
      const output = await page.$eval(
        "#output",
        (element) => element.textContent,
      );
      throw new Error(
        `example never printed ${JSON.stringify(marker)} within ${timeout}ms ` +
          `(${error.name}); #output was:\n${output}`,
      );
    }
  };

  await page.click("#run-button");
  await waitForExampleOutput("gradient: [2.0, 4.0, 6.0]", 30_000);
  await page.waitForFunction(
    () => !document.querySelector("#run-button")?.disabled,
    { timeout: 30_000 },
  );
  await page.select("#example-select", "transformersTiny");
  await page.click("#run-button");
  await waitForExampleOutput("WebGPU forward: passed", 180_000);
  const transformersOutput = await page.$eval(
    "#output",
    (element) => element.textContent,
  );
  if (
    !transformersOutput.includes("tokenizers: unavailable") ||
    !transformersOutput.includes("model: Qwen2ForCausalLM") ||
    !transformersOutput.includes("WebGPU forward: passed") ||
    !transformersOutput.includes("CPU fallbacks: 0") ||
    !transformersOutput.includes(`transformers: ${transformersPackage.version}`) ||
    !transformersOutput.includes(
      "WebGPU RMSNorm adapter: transformers-4.46.3-webgpu-rms-norm " +
        "(4 pinned classes)",
    ) ||
    !transformersOutput.includes(
      "WebGPU rotary adapter: transformers-4.46.3-webgpu-rotary-scaling " +
        "(2 pinned classes)",
    ) ||
    !transformersOutput.includes(
      "WebGPU decode SwiGLU adapter: " +
        "transformers-4.46.3-webgpu-decode-swiglu (3 pinned classes)",
    ) ||
    !transformersOutput.includes(
      "WebGPU OPT SDPA-mask adapter: " +
        "transformers-4.46.3-webgpu-opt-sdpa-mask (1 pinned class)",
    ) ||
    !transformersOutput.includes(
      "WebGPU preallocated KV cache: " +
        "transformers-4.46.3-webgpu-preallocated-kv " +
        "(3 dispatches saved/layer/token)",
    )
  ) {
    throw new Error(`model-only Transformers example failed: ${transformersOutput}`);
  }
  const wheelRequests = [...requests.entries()]
    .filter(([pathname]) => pathname.endsWith(".whl"))
    .reduce((total, [, count]) => total + count, 0);
  const manifestRequests = requests.get("/runtime/build-manifest.json") ?? 0;
  if (manifestRequests < 1 || wheelRequests < 1) {
    throw new Error(
      `expected manifest and wheel requests, got ${manifestRequests}/${wheelRequests}`,
    );
  }
  if (browserErrors.length !== 0) {
    throw new Error(`browser errors: ${browserErrors.join("\n")}`);
  }
  if (huggingFaceRequests.length !== 0) {
    throw new Error(`unexpected Hugging Face network requests: ${huggingFaceRequests.join("\n")}`);
  }
  if (pythonPackageIndexRequests.length !== 0) {
    throw new Error(
      `unexpected Python package index requests: ${pythonPackageIndexRequests.join("\n")}`,
    );
  }
  console.log(
    JSON.stringify(
      {
        playground: "passed",
        chrome,
        release: runtime.release,
        torch: runtime.torch,
        manifest_requests: manifestRequests,
        wheel_requests: wheelRequests,
        executed_autograd_example: true,
        executed_transformers_model_example: true,
        transformers: transformersPackage.version,
        webgpu_rms_norm_adapter: "transformers-4.46.3-webgpu-rms-norm",
        webgpu_swiglu_adapter: "transformers-4.46.3-webgpu-decode-swiglu",
        webgpu_opt_adapter: "transformers-4.46.3-webgpu-opt-sdpa-mask",
        webgpu_preallocated_kv_cache:
          "transformers-4.46.3-webgpu-preallocated-kv",
        webgpu_q8_linear: "webgpu-q8-group128-v1-decode",
        webgpu_gemma2_rms_norm:
          "transformers-4.46.3-webgpu-gemma2-rms-norm",
        tokenizers_installed: false,
        huggingface_network_requests: 0,
        python_package_index_requests: 0,
      },
      null,
      2,
    ),
  );
} catch (error) {
  console.error(
    JSON.stringify(
      {
        playground: "failed",
        chrome,
        error: error instanceof Error ? error.stack : String(error),
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
