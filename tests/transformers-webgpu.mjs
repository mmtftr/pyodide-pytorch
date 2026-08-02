import fs from "node:fs";
import crypto from "node:crypto";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const MAX_ERROR_CHARACTERS = 8_000;
const REQUIRED_PYODIDE_PACKAGES = [
  "fsspec",
  "jinja2",
  "networkx",
  "numpy",
  "pyyaml",
  "regex",
  "requests",
  "safetensors",
  "sympy",
  "tqdm",
  "typing-extensions",
];
const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));

function existingFile(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return path.resolve(candidate);
    }
  }
  return undefined;
}

function safeError(error) {
  const detail = error instanceof Error ? error.stack : String(error);
  if (detail.length <= MAX_ERROR_CHARACTERS) {
    return detail;
  }
  return `${detail.slice(0, MAX_ERROR_CHARACTERS)}… [truncated]`;
}

const [
  wheelArgument,
  filelockArgument,
  transformersArgument,
  hubArgument,
  pyodideArgument,
  pageArgument,
] = process.argv.slice(2);
if (
  !wheelArgument ||
  !filelockArgument ||
  !transformersArgument ||
  !hubArgument ||
  !pyodideArgument
) {
  throw new Error(
    "usage: node tests/transformers-webgpu.mjs " +
      "WHEEL FILELOCK_WHEEL TRANSFORMERS_WHEEL HUB_WHEEL " +
      "PYODIDE_DIST [TEST_PAGE]",
  );
}

const wheel = path.resolve(wheelArgument);
const filelockWheel = path.resolve(filelockArgument);
const transformersWheel = path.resolve(transformersArgument);
const hubWheel = path.resolve(hubArgument);
if (
  path.dirname(filelockWheel) !== path.dirname(transformersWheel) ||
  path.dirname(transformersWheel) !== path.dirname(hubWheel)
) {
  throw new Error("Transformers companion wheels must share one verified directory");
}
const browserManifest = path.join(
  path.dirname(transformersWheel),
  "transformers-browser-manifest.json",
);
const pyodideDirectory = path.resolve(pyodideArgument);
const rmsNormOnlyEnvironment = process.env.TRANSFORMERS_RMS_NORM_ONLY;
if (
  rmsNormOnlyEnvironment !== undefined &&
  !new Set(["0", "1"]).has(rmsNormOnlyEnvironment)
) {
  throw new Error("TRANSFORMERS_RMS_NORM_ONLY must be 0 or 1");
}
const rmsNormOnly = rmsNormOnlyEnvironment === "1";
const testPage = pageArgument
  ? path.resolve(pageArgument)
  : path.join(scriptDirectory, "transformers-webgpu.html");
const fixture = path.join(
  scriptDirectory,
  "fixtures",
  "transformers_tiny.json",
);
const bootstrap = path.join(
  scriptDirectory,
  "..",
  "site",
  "transformers_browser_bootstrap.py",
);
const q8Helper = path.join(
  scriptDirectory,
  "..",
  "site",
  "transformers_q8.py",
);
const gemma2Helper = path.join(
  scriptDirectory,
  "..",
  "site",
  "transformers_gemma2_webgpu.py",
);
const pyodideLock = path.join(pyodideDirectory, "pyodide-lock.json");
const adapterMode = process.env.WEBGPU_ADAPTER ?? "swiftshader";
if (!new Set(["hardware", "swiftshader"]).has(adapterMode)) {
  throw new Error(`unsupported WEBGPU_ADAPTER mode: ${adapterMode}`);
}
const chrome = existingFile([
  process.env.CHROME_PATH,
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]);
for (const [label, filename] of [
  ["wheel", wheel],
  ["filelock wheel", filelockWheel],
  ["Transformers wheel", transformersWheel],
  ["Hugging Face Hub wheel", hubWheel],
  ["Transformers browser manifest", browserManifest],
  ["Pyodide package", path.join(pyodideDirectory, "pyodide.mjs")],
  ["Pyodide lock file", pyodideLock],
  ["test page", testPage],
  ["tiny-model fixture", fixture],
  ["model-only bootstrap", bootstrap],
  ["Q8 conversion helper", q8Helper],
  ["Gemma2 WebGPU helper", gemma2Helper],
  ["Chrome", chrome],
]) {
  if (!filename || !fs.existsSync(filename)) {
    throw new Error(`${label} does not exist: ${filename}`);
  }
}

const manifest = JSON.parse(fs.readFileSync(browserManifest, "utf8"));
if (
  manifest.schema_version !== 1 ||
  manifest.requirements !== "config/transformers-browser-requirements.txt" ||
  manifest.model_only !== true ||
  manifest.tokenizers_included !== false ||
  !Array.isArray(manifest.packages)
) {
  throw new Error("invalid Transformers browser manifest");
}
const normalizePackageName = (name) => name.toLowerCase().replace(/[-_.]+/g, "-");
const manifestPackages = new Map(
  manifest.packages.map((entry) => [normalizePackageName(entry.name), entry]),
);
for (const [name, wheelPath] of [
  ["filelock", filelockWheel],
  ["transformers", transformersWheel],
  ["huggingface-hub", hubWheel],
]) {
  const entry = manifestPackages.get(name);
  if (entry?.filename !== path.basename(wheelPath)) {
    throw new Error(`manifest does not bind the supplied ${name} wheel`);
  }
  const bytes = fs.readFileSync(wheelPath);
  const digest = crypto.createHash("sha256").update(bytes).digest("hex");
  if (entry.size !== bytes.byteLength || entry.sha256 !== digest) {
    throw new Error(`manifest integrity check failed for ${name}`);
  }
}
if (manifestPackages.has("tokenizers")) {
  throw new Error("model-only manifest unexpectedly contains tokenizers");
}

const pyodidePackages = JSON.parse(
  fs.readFileSync(pyodideLock, "utf8"),
).packages;
const pendingPackages = [...REQUIRED_PYODIDE_PACKAGES];
const requiredPackageFiles = new Set();
const visitedPackages = new Set();
while (pendingPackages.length !== 0) {
  const packageName = pendingPackages.pop();
  if (visitedPackages.has(packageName)) {
    continue;
  }
  visitedPackages.add(packageName);
  const packageMetadata = pyodidePackages[packageName];
  if (packageMetadata === undefined) {
    throw new Error(`Pyodide lock file has no package named ${packageName}`);
  }
  const packageFile = packageMetadata.file_name;
  if (path.basename(packageFile) !== packageFile) {
    throw new Error(
      `Pyodide package ${packageName} has a non-local file: ${packageFile}`,
    );
  }
  requiredPackageFiles.add(packageFile);
  pendingPackages.push(...packageMetadata.depends);
}
const missingPackageFiles = [...requiredPackageFiles].filter(
  (packageFile) => !fs.existsSync(path.join(pyodideDirectory, packageFile)),
);
if (missingPackageFiles.length !== 0) {
  throw new Error(
    "Pyodide distribution is missing browser dependency files: " +
      missingPackageFiles.sort().join(", "),
  );
}

const contentTypes = new Map([
  [".data", "application/octet-stream"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".py", "text/x-python; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".whl", "application/zip"],
  [".zip", "application/zip"],
]);
const routes = new Map([
  ["/", testPage],
  ["/fixtures/transformers_tiny.json", fixture],
  ["/fixtures/transformers-browser-manifest.json", browserManifest],
  // Serve the checked-in Pages-root runtime module at an explicit local-only
  // harness URL. Both routes resolve to the same source file.
  ["/runtime/transformers_browser_bootstrap.py", bootstrap],
  ["/runtime/transformers_q8.py", q8Helper],
  ["/runtime/transformers_gemma2_webgpu.py", gemma2Helper],
  [`/dependency/${path.basename(filelockWheel)}`, filelockWheel],
  [`/dependency/${path.basename(transformersWheel)}`, transformersWheel],
  [`/dependency/${path.basename(hubWheel)}`, hubWheel],
  [`/wheel/${path.basename(wheel)}`, wheel],
]);

const server = http.createServer((request, response) => {
  const url = new URL(request.url, "http://127.0.0.1");
  let filename = routes.get(url.pathname);
  if (filename === undefined && url.pathname.startsWith("/pyodide/")) {
    const relative = url.pathname.slice("/pyodide/".length);
    const candidate = path.resolve(pyodideDirectory, relative);
    if (candidate.startsWith(`${pyodideDirectory}${path.sep}`)) {
      filename = candidate;
    }
  }
  if (filename === undefined || !fs.existsSync(filename)) {
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
  const localOrigin = `http://127.0.0.1:${address.port}`;
  const browserArguments = [
    "--no-sandbox",
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
  page.on("console", (message) => {
    console.log(`transformers browser: ${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    console.error(`transformers browser page error: ${safeError(error)}`);
  });
  await page.setRequestInterception(true);
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin !== localOrigin) {
      request.abort("blockedbyclient");
      return;
    }
    request.continue();
  });
  const parameters = new URLSearchParams({
    wheel: path.basename(wheel),
    filelock: path.basename(filelockWheel),
    transformers: path.basename(transformersWheel),
    hub: path.basename(hubWheel),
    adapter: adapterMode,
    rms_norm_only: rmsNormOnly ? "1" : "0",
  });
  await page.goto(`${localOrigin}/?${parameters.toString()}`);
  await page.waitForFunction(() => window.webgpuTest?.done, {
    timeout: 600_000,
  });
  const result = await page.evaluate(() => window.webgpuTest);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) {
    throw new Error(result.message);
  }
} catch (error) {
  console.error(
    JSON.stringify(
      {
        transformersWebGPU: "failed",
        chrome,
        wheel,
        rmsNormOnly,
        error: safeError(error),
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
} finally {
  if (browser !== undefined) {
    await browser.close();
  }
  await new Promise((resolve) => server.close(resolve));
}
