import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const [wheelArgument, filelockArgument, pyodideArgument, pageArgument] =
  process.argv.slice(2);
if (!wheelArgument || !filelockArgument || !pyodideArgument) {
  throw new Error(
    "usage: node tests/serve-webgpu.mjs WHEEL FILELOCK_WHEEL PYODIDE_DIST [TEST_PAGE]",
  );
}

const wheel = path.resolve(wheelArgument);
const filelockWheel = path.resolve(filelockArgument);
const pyodideDirectory = path.resolve(pyodideArgument);
const testPage = path.resolve(pageArgument ?? "tests/webgpu.html");
for (const [label, filename] of [
  ["wheel", wheel],
  ["filelock wheel", filelockWheel],
  ["Pyodide package", path.join(pyodideDirectory, "pyodide.mjs")],
  ["test page", testPage],
]) {
  if (!fs.existsSync(filename)) {
    throw new Error(`${label} does not exist: ${filename}`);
  }
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
  ["/", testPage],
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

const port = Number.parseInt(process.env.WEBGPU_PORT ?? "8765", 10);
await new Promise((resolve, reject) => {
  server.once("error", reject);
  server.listen(port, "127.0.0.1", resolve);
});
const parameters = new URLSearchParams({
  wheel: path.basename(wheel),
  filelock: path.basename(filelockWheel),
    adapter: process.env.WEBGPU_ADAPTER ?? "hardware",
    timestamps: process.env.WEBGPU_TIMESTAMPS ?? "0",
    extended: process.env.WEBGPU_EXTENDED ?? "0",
    batch: process.env.WEBGPU_BATCH ?? "0",
  });
console.log(`http://127.0.0.1:${port}/?${parameters.toString()}`);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
