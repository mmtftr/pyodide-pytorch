import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const shader = path.resolve(
  scriptDirectory,
  "..",
  "webgpu",
  "llm_kernels",
  "scalar_binary.wgsl",
);

function existingFile(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return path.resolve(candidate);
  }
  return undefined;
}

const chrome = existingFile([
  process.env.CHROME_PATH,
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]);
if (!chrome) throw new Error("Chrome executable was not found");
if (!fs.existsSync(shader)) throw new Error(`shader does not exist: ${shader}`);

const pageSource = String.raw`<!doctype html>
<meta charset="utf-8">
<script type="module">
const result = { done: false, ok: false, errors: [] };
globalThis.scalarBinaryTest = result;

function gpuBuffer(device, label, values, usage) {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, values.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  if (values.byteLength !== 0) device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function floatBits(value) {
  const bytes = new ArrayBuffer(4);
  new Float32Array(bytes)[0] = value;
  return new Uint32Array(bytes)[0];
}

function paramsBuffer(device, testCase) {
  const words = new Uint32Array(32);
  words.set([
    6,
    2,
    9,
    1,
    floatBits(testCase.scalar),
    floatBits(testCase.alpha),
    testCase.operation,
    1,
  ]);
  words.set([2, 3], 8);
  words.set([8, 2], 16);
  words.set([1, 2], 24);
  return gpuBuffer(
    device,
    testCase.name + " params",
    words,
    GPUBufferUsage.UNIFORM,
  );
}

async function execute(device, pipeline, input, testCase) {
  const outputInitial = new Float32Array(8).fill(777);
  const output = gpuBuffer(
    device,
    testCase.name + " output",
    outputInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const params = paramsBuffer(device, testCase);
  const readback = device.createBuffer({
    label: testCase.name + " readback",
    size: outputInitial.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input.buffer } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: params } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(
    output,
    0,
    readback,
    0,
    outputInitial.byteLength,
  );
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const storage = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  // Output strides (1, 2) deliberately differ from contiguous strides. The
  // first and last storage words are sentinels outside the logical view.
  const actual = [storage[1], storage[3], storage[5],
                  storage[2], storage[4], storage[6]];
  actual.forEach((value, index) => {
    const expected = testCase.reference(input.logical[index]);
    const tolerance = 1e-5 * Math.max(1, Math.abs(expected));
    if (Math.abs(value - expected) > tolerance) {
      throw new Error(
        testCase.name + " index " + index + ": " + value + " != " + expected,
      );
    }
  });
  if (storage[0] !== 777 || storage[7] !== 777) {
    throw new Error(testCase.name + " wrote outside its strided output view");
  }
  output.destroy();
  params.destroy();
  readback.destroy();
  return actual;
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  device.addEventListener("uncapturederror", (event) => {
    result.errors.push(event.error?.message ?? String(event.error));
  });
  const source = await (await fetch("/scalar_binary.wgsl")).text();
  const module = device.createShaderModule({ code: source });
  const compilation = await module.getCompilationInfo();
  const compilationErrors = compilation.messages.filter(
    (message) => message.type === "error",
  );
  if (compilationErrors.length !== 0) {
    throw new Error(compilationErrors.map((message) => message.message).join("\n"));
  }
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const values = Float32Array.from({ length: 24 }, (_, index) => index - 12);
  const input = {
    buffer: gpuBuffer(device, "strided input", values, GPUBufferUsage.STORAGE),
    logical: [values[9], values[11], values[13], values[17], values[19], values[21]],
  };
  const cases = [
    {
      name: "add-epsilon",
      operation: 0,
      scalar: 1e-6,
      alpha: 1,
      reference: (value) => value + 1e-6,
    },
    {
      name: "sub-alpha",
      operation: 1,
      scalar: 0.75,
      alpha: 2,
      reference: (value) => value - 1.5,
    },
    {
      name: "multiply",
      operation: 2,
      scalar: -3,
      alpha: 1,
      reference: (value) => value * -3,
    },
    {
      name: "divide",
      operation: 3,
      scalar: 2.5,
      alpha: 1,
      reference: (value) => value / 2.5,
    },
  ];
  result.cases = {};
  for (const testCase of cases) {
    result.cases[testCase.name] = await execute(
      device,
      pipeline,
      input,
      testCase,
    );
  }
  input.buffer.destroy();
  const validationError = await device.popErrorScope();
  if (validationError) result.errors.push(validationError.message);
  if (result.errors.length !== 0) throw new Error(result.errors.join("\n"));
  result.ok = true;
} catch (error) {
  result.message = error instanceof Error ? error.stack : String(error);
} finally {
  result.done = true;
}
</script>`;

const server = http.createServer((request, response) => {
  if (request.url === "/scalar_binary.wgsl") {
    response.setHeader("Content-Type", "text/plain; charset=utf-8");
    fs.createReadStream(shader).pipe(response);
    return;
  }
  response.setHeader("Content-Type", "text/html; charset=utf-8");
  response.end(pageSource);
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
      "--use-gpu-in-tests",
    ],
  });
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${address.port}/`);
  await page.waitForFunction(() => globalThis.scalarBinaryTest?.done, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => globalThis.scalarBinaryTest);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) throw new Error(result.message ?? result.errors.join("\n"));
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
