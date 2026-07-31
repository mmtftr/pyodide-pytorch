import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const MAX_ERROR_CHARACTERS = 4_000;
const REQUIRED_PYODIDE_PACKAGES = [
  "numpy",
  "typing-extensions",
  "sympy",
  "networkx",
  "jinja2",
  "fsspec",
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

const [wheelArgument, filelockArgument, pyodideArgument] =
  process.argv.slice(2);
if (!wheelArgument || !filelockArgument || !pyodideArgument) {
  throw new Error(
    "usage: node tests/webgpu.mjs WHEEL FILELOCK_WHEEL PYODIDE_DIST",
  );
}
const wheel = path.resolve(wheelArgument);
const filelockWheel = path.resolve(filelockArgument);
const pyodideDirectory = path.resolve(pyodideArgument);
const pyodideLock = path.join(pyodideDirectory, "pyodide-lock.json");
const chrome = existingFile([
  process.env.CHROME_PATH,
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
]);
for (const [label, filename] of [
  ["wheel", wheel],
  ["filelock wheel", filelockWheel],
  ["Pyodide package", path.join(pyodideDirectory, "pyodide.mjs")],
  ["Pyodide lock file", pyodideLock],
  ["Chrome", chrome],
]) {
  if (!filename || !fs.existsSync(filename)) {
    throw new Error(`${label} does not exist: ${filename}`);
  }
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
  [".wasm", "application/wasm"],
  [".whl", "application/zip"],
  [".zip", "application/zip"],
]);
const routes = new Map([
  ["/", path.join(scriptDirectory, "webgpu.html")],
  [`/dependency/${path.basename(filelockWheel)}`, filelockWheel],
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
  browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args: [
      "--no-sandbox",
      "--enable-unsafe-webgpu",
      "--enable-unsafe-swiftshader",
      "--use-webgpu-adapter=swiftshader",
      "--enable-dawn-features=allow_unsafe_apis",
      "--disable-dawn-features=use_dxc",
      "--enable-webgpu-developer-features",
      "--use-gpu-in-tests",
      "--enable-accelerated-2d-canvas",
    ],
  });
  const page = await browser.newPage();
  page.on("console", (message) => {
    console.log(`webgpu browser: ${message.type()}: ${message.text()}`);
  });
  page.on("pageerror", (error) => {
    console.error(`webgpu browser page error: ${safeError(error)}`);
  });
  const parameters = new URLSearchParams({
    wheel: path.basename(wheel),
    filelock: path.basename(filelockWheel),
  });
  await page.goto(
    `http://127.0.0.1:${address.port}/?${parameters.toString()}`,
  );
  await page.waitForFunction(() => window.webgpuTest?.done, {
    timeout: 300_000,
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
        webgpu: "failed",
        chrome,
        wheel,
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
