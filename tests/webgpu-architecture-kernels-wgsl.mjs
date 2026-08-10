import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const kernelDirectory = path.resolve(
  scriptDirectory,
  "..",
  "webgpu",
  "llm_kernels",
);
const shaders = new Map([
  ["/long_arithmetic.wgsl", path.join(kernelDirectory, "long_arithmetic.wgsl")],
  ["/mixed_pow.wgsl", path.join(kernelDirectory, "mixed_pow.wgsl")],
  ["/baddbmm.wgsl", path.join(kernelDirectory, "baddbmm.wgsl")],
]);

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
for (const filename of shaders.values()) {
  if (!fs.existsSync(filename)) throw new Error(`shader does not exist: ${filename}`);
}

const pageSource = String.raw`<!doctype html>
<meta charset="utf-8">
<script type="module">
const result = { done: false, ok: false, errors: [], cases: {} };
globalThis.architectureKernelTest = result;

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

function longWords(values) {
  const words = new Uint32Array(values.length * 2);
  values.forEach((value, index) => {
    words[index * 2] = value >>> 0;
    words[index * 2 + 1] = value < 0 ? 0xffffffff : 0;
  });
  return words;
}

function assertClose(name, actual, expected, tolerance = 2e-5) {
  if (actual.length !== expected.length) {
    throw new Error(name + " length mismatch");
  }
  actual.forEach((value, index) => {
    const bound = tolerance * Math.max(1, Math.abs(expected[index]));
    if (!Number.isFinite(value) || Math.abs(value - expected[index]) > bound) {
      throw new Error(
        name + " index " + index + ": " + value + " != " + expected[index],
      );
    }
  });
}

async function pipeline(device, url) {
  const source = await (await fetch(url)).text();
  const module = device.createShaderModule({ label: url, code: source });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
  if (errors.length !== 0) {
    throw new Error(url + ": " + errors.map((message) => message.message).join("\n"));
  }
  return device.createComputePipelineAsync({
    label: url,
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

async function readBuffer(device, encoder, source, byteLength, Type) {
  const readback = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Type(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return [...values];
}

function longParams(operation, rhsIsScalar, scalar = 0, alpha = 1) {
  const words = new Uint32Array(52);
  words.set([
    6, 2, 2, rhsIsScalar ? 0 : 1,
    0, 0, operation, alpha >>> 0,
    scalar >>> 0, scalar < 0 ? 0xffffffff : 0,
    rhsIsScalar ? 1 : 0, 1,
  ]);
  words.set([2, 3], 12);
  words.set([2, 3], 20);
  words.set([3, 1], 28);
  if (!rhsIsScalar) {
    words.set([3], 36);
    words.set([1], 44);
  }
  return words;
}

async function runLong(device, arithmeticPipeline) {
  const lhsValues = [-4, -1, 0, 2, 7, 10];
  const rhsValues = [2, -3, 4];
  const lhs = gpuBuffer(device, "Long lhs", longWords(lhsValues), GPUBufferUsage.STORAGE);
  const rhs = gpuBuffer(device, "Long rhs", longWords(rhsValues), GPUBufferUsage.STORAGE);
  const cases = [
    { name: "add.Tensor alpha", op: 0, alpha: 2, expected: [0, -7, 8, 6, 1, 18] },
    { name: "sub.Tensor", op: 1, expected: [-6, 2, -4, 0, 10, 6] },
    { name: "mul.Tensor", op: 2, expected: [-8, 3, 0, 4, -21, 40] },
    { name: "minimum", op: 3, expected: [-4, -3, 0, 2, -3, 4] },
    { name: "rsub.Scalar", op: 4, scalar: 10, alpha: 2, scalarCase: true,
      expected: [18, 12, 10, 6, -4, -10] },
    { name: "neg", op: 5, scalarCase: true, expected: [4, 1, 0, -2, -7, -10] },
    { name: "abs", op: 6, scalarCase: true, expected: [4, 1, 0, 2, 7, 10] },
  ];
  for (const testCase of cases) {
    const output = gpuBuffer(
      device,
      testCase.name + " output",
      new Uint32Array(12),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const params = gpuBuffer(
      device,
      testCase.name + " params",
      longParams(
        testCase.op,
        Boolean(testCase.scalarCase),
        testCase.scalar ?? 0,
        testCase.alpha ?? 1,
      ),
      GPUBufferUsage.UNIFORM,
    );
    const bindGroup = device.createBindGroup({
      layout: arithmeticPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: lhs } },
        { binding: 1, resource: { buffer: testCase.scalarCase ? lhs : rhs } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(arithmeticPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    const actual = await readBuffer(device, encoder, output, 48, Uint32Array);
    const expected = [...longWords(testCase.expected)];
    if (actual.some((value, index) => value !== expected[index])) {
      throw new Error(testCase.name + " produced " + actual + ", expected " + expected);
    }
    result.cases[testCase.name] = testCase.expected;
    output.destroy();
    params.destroy();
  }
  lhs.destroy();
  rhs.destroy();
}

async function runMixedPow(device, mixedPowPipeline) {
  const baseValues = [2, 4, 8, 3, 9, 27];
  const exponentValues = [1, -1, 2];
  const base = gpuBuffer(
    device, "pow base", new Float32Array(baseValues), GPUBufferUsage.STORAGE,
  );
  const exponent = gpuBuffer(
    device, "pow exponent", longWords(exponentValues), GPUBufferUsage.STORAGE,
  );
  const output = gpuBuffer(
    device, "pow output", new Float32Array(6),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const words = new Uint32Array(48);
  words.set([6, 2, 2, 1, 0, 0, 2, 1]);
  words.set([2, 3], 8);
  words.set([2, 3], 16);
  words.set([3, 1], 24);
  words.set([3], 32);
  words.set([1], 40);
  const params = gpuBuffer(device, "pow params", words, GPUBufferUsage.UNIFORM);
  const bindGroup = device.createBindGroup({
    layout: mixedPowPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: base } },
      { binding: 1, resource: { buffer: exponent } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: params } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(mixedPowPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  const actual = await readBuffer(device, encoder, output, 24, Float32Array);
  const expected = [2, 0.25, 64, 3, 1 / 9, 729];
  assertClose("pow.Tensor_Tensor", actual, expected);
  result.cases["pow.Tensor_Tensor"] = actual;
  for (const buffer of [base, exponent, output, params]) buffer.destroy();
}

async function runBaddbmm(device, baddbmmPipeline) {
  const lhsValues = [1, 2, 3, 4, 5, 6, -1, 2, 0, 3, -2, 1];
  const rhsValues = [1, 0, 0, 1, 1, 1, 2, -1, 1, 3, -2, 4];
  const selfValues = [0.5, -2];
  const lhs = gpuBuffer(device, "baddbmm lhs", new Float32Array(lhsValues), GPUBufferUsage.STORAGE);
  const rhs = gpuBuffer(device, "baddbmm rhs", new Float32Array(rhsValues), GPUBufferUsage.STORAGE);
  const self = gpuBuffer(device, "baddbmm self", new Float32Array(selfValues), GPUBufferUsage.STORAGE);
  const output = gpuBuffer(
    device, "baddbmm output", new Float32Array(8),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const words = new Uint32Array(28);
  words.set([2, 2, 2, 3, 1, 0, 0, 0, 0, floatBits(0.5), floatBits(-1.25), 0]);
  words.set([2], 12);
  words.set([1], 16);
  words.set([6, 3, 1], 20);
  words.set([6, 2, 1], 24);
  const params = gpuBuffer(device, "baddbmm params", words, GPUBufferUsage.UNIFORM);
  const bindGroup = device.createBindGroup({
    layout: baddbmmPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: self } },
      { binding: 1, resource: { buffer: lhs } },
      { binding: 2, resource: { buffer: rhs } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: params } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(baddbmmPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1, 1, 2);
  pass.end();
  const actual = await readBuffer(device, encoder, output, 32, Float32Array);
  const expected = [];
  for (let batch = 0; batch < 2; batch++) {
    for (let row = 0; row < 2; row++) {
      for (let column = 0; column < 2; column++) {
        let dot = 0;
        for (let inner = 0; inner < 3; inner++) {
          dot += lhsValues[batch * 6 + row * 3 + inner] *
            rhsValues[batch * 6 + inner * 2 + column];
        }
        expected.push(-1.25 * selfValues[column] + 0.5 * dot);
      }
    }
  }
  assertClose("baddbmm", actual, expected);
  result.cases.baddbmm = actual;

  device.queue.writeBuffer(self, 0, new Float32Array([NaN, NaN]));
  words[10] = floatBits(0);
  device.queue.writeBuffer(params, 0, words);
  const betaZeroEncoder = device.createCommandEncoder();
  const betaZeroPass = betaZeroEncoder.beginComputePass();
  betaZeroPass.setPipeline(baddbmmPipeline);
  betaZeroPass.setBindGroup(0, bindGroup);
  betaZeroPass.dispatchWorkgroups(1, 1, 2);
  betaZeroPass.end();
  const betaZeroActual = await readBuffer(
    device,
    betaZeroEncoder,
    output,
    32,
    Float32Array,
  );
  const betaZeroExpected = expected.map((value, index) =>
    value + 1.25 * selfValues[index % 2],
  );
  assertClose("baddbmm beta zero ignores NaN self", betaZeroActual, betaZeroExpected);
  result.cases["baddbmm beta zero ignores NaN self"] = betaZeroActual;
  for (const buffer of [lhs, rhs, self, output, params]) buffer.destroy();
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  device.addEventListener("uncapturederror", (event) => {
    result.errors.push(event.error?.message ?? String(event.error));
  });
  const [longPipeline, powPipeline, baddbmmPipeline] = await Promise.all([
    pipeline(device, "/long_arithmetic.wgsl"),
    pipeline(device, "/mixed_pow.wgsl"),
    pipeline(device, "/baddbmm.wgsl"),
  ]);
  await runLong(device, longPipeline);
  await runMixedPow(device, powPipeline);
  await runBaddbmm(device, baddbmmPipeline);
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
  const filename = shaders.get(request.url);
  if (filename) {
    response.setHeader("Content-Type", "text/plain; charset=utf-8");
    fs.createReadStream(filename).pipe(response);
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
  await page.waitForFunction(() => globalThis.architectureKernelTest?.done, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => globalThis.architectureKernelTest);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) throw new Error(result.message ?? result.errors.join("\n"));
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
