import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const shaderDirectory = path.resolve(
  scriptDirectory,
  "..",
  "webgpu",
  "llm_kernels",
);
const shaderNames = [
  "all_bool.wgsl",
  "eq_scalar.wgsl",
  "fill.wgsl",
  "int_to_float.wgsl",
  "strided_copy.wgsl",
];

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
for (const name of shaderNames) {
  const filename = path.join(shaderDirectory, name);
  if (!fs.existsSync(filename)) throw new Error(`shader does not exist: ${filename}`);
}

const pageSource = String.raw`<!doctype html>
<meta charset="utf-8">
<script type="module">
const result = { done: false, ok: false, errors: [] };
globalThis.boolShaderTest = result;

function gpuBuffer(device, label, values, usage) {
  const size = Math.max(4, (values.byteLength + 3) & ~3);
  const buffer = device.createBuffer({
    label,
    size,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  if (values.byteLength !== 0) device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

async function pipeline(device, name) {
  const source = await (await fetch("/" + name)).text();
  const module = device.createShaderModule({ label: name, code: source });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(name + ": " + errors.map((message) => message.message).join("\n"));
  }
  return device.createComputePipelineAsync({
    label: name,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

function bindGroup(device, pipelineValue, buffers) {
  return device.createBindGroup({
    layout: pipelineValue.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    })),
  });
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  device.addEventListener("uncapturederror", (event) => {
    result.errors.push(event.error?.message ?? String(event.error));
  });

  const pipelines = Object.fromEntries(
    await Promise.all(
      ${JSON.stringify(shaderNames)}.map(async (name) => [
        name,
        await pipeline(device, name),
      ]),
    ),
  );

  // Eight canonical restricted-Long values. eq.Scalar should produce eight
  // one-byte Bool values packed into two words, then all should reduce false.
  const longWords = new Uint32Array([
    1, 0, 1, 0, 0, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 1, 0,
  ]);
  const longInput = gpuBuffer(
    device,
    "Long comparison input",
    longWords,
    GPUBufferUsage.STORAGE,
  );
  const comparisonOutput = device.createBuffer({
    label: "packed Bool comparison output",
    size: 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const eqParamsBytes = new ArrayBuffer(96);
  const eqParams = new Uint32Array(eqParamsBytes);
  eqParams[0] = 8; // length
  eqParams[1] = 1; // ndim
  eqParams[2] = 0; // input offset
  eqParams[3] = 2; // restricted Long
  eqParams[4] = 1; // scalar low word
  eqParams[5] = 0; // scalar high word
  eqParams[6] = 1; // dispatch x
  eqParams[8] = 8; // sizes[0]
  eqParams[16] = 1; // strides[0]
  const eqParamsBuffer = gpuBuffer(
    device,
    "eq params",
    new Uint8Array(eqParamsBytes),
    GPUBufferUsage.UNIFORM,
  );

  const allOutput = device.createBuffer({
    label: "Bool all output",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const allParamsBytes = new ArrayBuffer(80);
  const allParams = new Uint32Array(allParamsBytes);
  allParams[0] = 8;
  allParams[1] = 1;
  allParams[2] = 0;
  allParams[4] = 8;
  allParams[12] = 1;
  const allParamsBuffer = gpuBuffer(
    device,
    "all params",
    new Uint8Array(allParamsBytes),
    GPUBufferUsage.UNIFORM,
  );

  const readback = device.createBuffer({
    label: "Bool comparison readback",
    size: 12,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  let pass = encoder.beginComputePass();
  pass.setPipeline(pipelines["eq_scalar.wgsl"]);
  pass.setBindGroup(
    0,
    bindGroup(device, pipelines["eq_scalar.wgsl"], [
      longInput,
      comparisonOutput,
      eqParamsBuffer,
    ]),
  );
  pass.dispatchWorkgroups(1);
  pass.end();
  pass = encoder.beginComputePass();
  pass.setPipeline(pipelines["all_bool.wgsl"]);
  pass.setBindGroup(
    0,
    bindGroup(device, pipelines["all_bool.wgsl"], [
      comparisonOutput,
      allOutput,
      allParamsBuffer,
    ]),
  );
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(comparisonOutput, 0, readback, 0, 8);
  encoder.copyBufferToBuffer(allOutput, 0, readback, 8, 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const mapped = readback.getMappedRange().slice(0);
  readback.unmap();
  const comparison = [...new Uint8Array(mapped, 0, 8)];
  const expectedComparison = [1, 1, 0, 1, 1, 0, 1, 1];
  if (comparison.some((value, index) => value !== expectedComparison[index])) {
    throw new Error("packed eq.Scalar mismatch: " + comparison);
  }
  const allValue = new Uint32Array(mapped, 8, 1)[0];
  if (allValue !== 0) throw new Error("Bool all mismatch: " + allValue);

  // Fill seven bytes at byte offset one. Boundary bytes in the two touched
  // words must be preserved, proving word-owned read/modify/write behavior.
  const fillInitial = new Uint8Array(12).fill(0xaa);
  const fillOutput = gpuBuffer(
    device,
    "Bool fill output",
    fillInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const fillParamsBytes = new ArrayBuffer(32);
  const fillParams = new Uint32Array(fillParamsBytes);
  fillParams[0] = 2; // touched words
  fillParams[1] = 1; // dispatch x
  fillParams[2] = 1; // byte offset
  fillParams[3] = 0; // Bool sentinel
  fillParams[4] = 1; // true byte
  fillParams[5] = 7; // logical length
  const fillParamsBuffer = gpuBuffer(
    device,
    "fill params",
    new Uint8Array(fillParamsBytes),
    GPUBufferUsage.UNIFORM,
  );
  const fillReadback = device.createBuffer({
    label: "Bool fill readback",
    size: 12,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const fillEncoder = device.createCommandEncoder();
  const fillPass = fillEncoder.beginComputePass();
  fillPass.setPipeline(pipelines["fill.wgsl"]);
  fillPass.setBindGroup(
    0,
    bindGroup(device, pipelines["fill.wgsl"], [fillOutput, fillParamsBuffer]),
  );
  fillPass.dispatchWorkgroups(1);
  fillPass.end();
  fillEncoder.copyBufferToBuffer(fillOutput, 0, fillReadback, 0, 12);
  device.queue.submit([fillEncoder.finish()]);
  await fillReadback.mapAsync(GPUMapMode.READ);
  const filled = [...new Uint8Array(fillReadback.getMappedRange().slice(0))];
  fillReadback.unmap();
  const expectedFill = [0xaa, 1, 1, 1, 1, 1, 1, 1, 0xaa, 0xaa, 0xaa, 0xaa];
  if (filled.some((value, index) => value !== expectedFill[index])) {
    throw new Error("packed Bool fill mismatch: " + filled);
  }

  // Materialize a rank-four offset prefix view with capacity-sized gaps
  // between heads. This is the storage pattern returned by a preallocated KV
  // cache, and is the GPU step used before its one asynchronous CPU readback.
  const prefixSourceValues = Float32Array.from(
    { length: 120 },
    (_, index) => index + 0.25,
  );
  const prefixSource = gpuBuffer(
    device,
    "strided copy offset prefix source",
    prefixSourceValues,
    GPUBufferUsage.STORAGE,
  );
  const prefixOutputInitial = new Float32Array(40).fill(-777.5);
  const prefixOutput = gpuBuffer(
    device,
    "strided copy offset prefix output",
    prefixOutputInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const prefixParamsBytes = new ArrayBuffer(128);
  const prefixParams = new Uint32Array(prefixParamsBytes);
  prefixParams[0] = 36; // length: [1, 3, 3, 4]
  prefixParams[1] = 4; // ndim
  prefixParams[2] = 64; // source offset: base[1, 0, 1, 0]
  prefixParams[3] = 0; // contiguous destination offset
  prefixParams[4] = 1; // one word per float32 element
  prefixParams.set([1, 3, 3, 4], 8);
  prefixParams.set([60, 20, 4, 1], 16);
  prefixParams.set([36, 12, 4, 1], 24);
  const prefixParamsBuffer = gpuBuffer(
    device,
    "strided copy offset prefix params",
    new Uint8Array(prefixParamsBytes),
    GPUBufferUsage.UNIFORM,
  );
  const prefixReadback = device.createBuffer({
    label: "strided copy offset prefix readback",
    size: prefixOutputInitial.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const prefixEncoder = device.createCommandEncoder();
  const prefixPass = prefixEncoder.beginComputePass();
  prefixPass.setPipeline(pipelines["strided_copy.wgsl"]);
  prefixPass.setBindGroup(
    0,
    bindGroup(device, pipelines["strided_copy.wgsl"], [
      prefixSource,
      prefixOutput,
      prefixParamsBuffer,
    ]),
  );
  prefixPass.dispatchWorkgroups(1);
  prefixPass.end();
  prefixEncoder.copyBufferToBuffer(
    prefixOutput,
    0,
    prefixReadback,
    0,
    prefixOutputInitial.byteLength,
  );
  device.queue.submit([prefixEncoder.finish()]);
  await prefixReadback.mapAsync(GPUMapMode.READ);
  const prefixActual = [
    ...new Float32Array(prefixReadback.getMappedRange().slice(0)),
  ];
  prefixReadback.unmap();
  const prefixExpected = [];
  for (let head = 0; head < 3; head++) {
    for (let token = 0; token < 3; token++) {
      for (let feature = 0; feature < 4; feature++) {
        prefixExpected.push(
          prefixSourceValues[64 + head * 20 + token * 4 + feature],
        );
      }
    }
  }
  prefixExpected.push(...new Array(4).fill(-777.5));
  if (
    prefixActual.some((value, index) => value !== prefixExpected[index])
  ) {
    throw new Error("strided copy offset prefix mismatch: " + prefixActual);
  }

  const validationError = await device.popErrorScope();
  if (validationError) result.errors.push(validationError.message);
  if (result.errors.length !== 0) throw new Error(result.errors.join("\n"));
  result.comparison = comparison;
  result.all = allValue;
  result.fill = filled;
  result.stridedCopy = prefixActual;
  result.compiled = ${JSON.stringify(shaderNames)};
  result.ok = true;
} catch (error) {
  result.message = error instanceof Error ? error.stack : String(error);
} finally {
  result.done = true;
}
</script>`;

const server = http.createServer((request, response) => {
  const name = request.url.slice(1);
  if (shaderNames.includes(name)) {
    response.setHeader("Content-Type", "text/plain; charset=utf-8");
    fs.createReadStream(path.join(shaderDirectory, name)).pipe(response);
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
  await page.waitForFunction(() => globalThis.boolShaderTest?.done, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => globalThis.boolShaderTest);
  if (!result.ok) throw new Error(result.message ?? result.errors.join("\n"));
  console.log(JSON.stringify({ boolWGSL: "passed", ...result }, null, 2));
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
