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
  "layer_norm.wgsl",
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
globalThis.layerNormTest = result;

function gpuBuffer(device, label, values, usage) {
  const size = Math.max(4, values.byteLength);
  const buffer = device.createBuffer({
    label,
    size,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  if (values.byteLength !== 0) device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function paramsBuffer(device, testCase) {
  const bytes = new ArrayBuffer(64);
  const words = new Uint32Array(bytes);
  words[0] = testCase.rows;
  words[1] = testCase.width;
  words[2] = testCase.inputOffset;
  words[3] = testCase.outputOffset;
  words[4] = testCase.meanOffset;
  words[5] = testCase.rstdOffset;
  words[6] = testCase.weightOffset;
  words[7] = testCase.biasOffset;
  words[8] = testCase.hasWeight ? 1 : 0;
  words[9] = testCase.hasBias ? 1 : 0;
  words[10] = testCase.dispatchX;
  new Float32Array(bytes)[12] = testCase.epsilon;
  return gpuBuffer(
    device,
    testCase.name + " params",
    new Uint8Array(bytes),
    GPUBufferUsage.UNIFORM,
  );
}

function reference(testCase) {
  const output = new Float32Array(testCase.rows * testCase.width);
  const mean = new Float64Array(testCase.rows);
  const rstd = new Float64Array(testCase.rows);
  for (let row = 0; row < testCase.rows; ++row) {
    const base = testCase.inputOffset + row * testCase.width;
    let rowMean = 0;
    for (let feature = 0; feature < testCase.width; ++feature) {
      rowMean += testCase.input[base + feature];
    }
    rowMean /= testCase.width;
    let m2 = 0;
    for (let feature = 0; feature < testCase.width; ++feature) {
      const delta = testCase.input[base + feature] - rowMean;
      m2 += delta * delta;
    }
    const rowRstd = 1 / Math.sqrt(m2 / testCase.width + testCase.epsilon);
    mean[row] = rowMean;
    rstd[row] = rowRstd;
    for (let feature = 0; feature < testCase.width; ++feature) {
      let value = (testCase.input[base + feature] - rowMean) * rowRstd;
      if (testCase.hasWeight) {
        value *= testCase.weight[testCase.weightOffset + feature];
      }
      if (testCase.hasBias) {
        value += testCase.bias[testCase.biasOffset + feature];
      }
      output[row * testCase.width + feature] = value;
    }
  }
  return { output, mean, rstd };
}

function maxError(actual, expected) {
  let error = 0;
  for (let index = 0; index < expected.length; ++index) {
    error = Math.max(error, Math.abs(actual[index] - expected[index]));
  }
  return error;
}

async function execute(device, pipeline, testCase) {
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const input = gpuBuffer(device, testCase.name + " input", testCase.input, storage);
  const weight = testCase.hasWeight
    ? gpuBuffer(device, testCase.name + " weight", testCase.weight, storage)
    : input;
  const bias = testCase.hasBias
    ? gpuBuffer(device, testCase.name + " bias", testCase.bias, storage)
    : input;
  const output = gpuBuffer(
    device,
    testCase.name + " output",
    new Float32Array(testCase.outputOffset + testCase.rows * testCase.width),
    storage,
  );
  const mean = gpuBuffer(
    device,
    testCase.name + " mean",
    new Float32Array(testCase.meanOffset + testCase.rows),
    storage,
  );
  const rstd = gpuBuffer(
    device,
    testCase.name + " rstd",
    new Float32Array(testCase.rstdOffset + testCase.rows),
    storage,
  );
  const params = paramsBuffer(device, testCase);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [input, weight, bias, output, mean, rstd, params].map(
      (buffer, binding) => ({ binding, resource: { buffer } }),
    ),
  });

  const outputBytes = testCase.rows * testCase.width * 4;
  const statisticBytes = testCase.rows * 4;
  const readback = device.createBuffer({
    label: testCase.name + " readback",
    size: outputBytes + 2 * statisticBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(
    testCase.dispatchX,
    Math.ceil(testCase.rows / testCase.dispatchX),
  );
  pass.end();
  encoder.copyBufferToBuffer(
    output,
    testCase.outputOffset * 4,
    readback,
    0,
    outputBytes,
  );
  encoder.copyBufferToBuffer(
    mean,
    testCase.meanOffset * 4,
    readback,
    outputBytes,
    statisticBytes,
  );
  encoder.copyBufferToBuffer(
    rstd,
    testCase.rstdOffset * 4,
    readback,
    outputBytes + statisticBytes,
    statisticBytes,
  );
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const expected = reference(testCase);
  const outputActual = values.subarray(0, testCase.rows * testCase.width);
  const meanActual = values.subarray(
    testCase.rows * testCase.width,
    testCase.rows * testCase.width + testCase.rows,
  );
  const rstdActual = values.subarray(
    testCase.rows * testCase.width + testCase.rows,
  );
  const errors = {
    output: maxError(outputActual, expected.output),
    mean: maxError(meanActual, expected.mean),
    rstd: maxError(rstdActual, expected.rstd),
  };
  for (const [field, error] of Object.entries(errors)) {
    if (!(error <= testCase.tolerance[field])) {
      throw new Error(
        testCase.name + " " + field + " error " + error +
        " exceeds " + testCase.tolerance[field],
      );
    }
  }
  return errors;
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  device.addEventListener("uncapturederror", (event) => {
    result.errors.push(event.error?.message ?? String(event.error));
  });
  const source = await (await fetch("/layer_norm.wgsl")).text();
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

  const stableRows = 3;
  const stableWidth = 259;
  const stableInputOffset = 3;
  const stableInput = new Float32Array(
    stableInputOffset + stableRows * stableWidth,
  );
  for (let row = 0; row < stableRows; ++row) {
    for (let feature = 0; feature < stableWidth; ++feature) {
      stableInput[stableInputOffset + row * stableWidth + feature] =
        8192 + row * 0.5 + ((feature * 13) % 29 - 14) * 0.125;
    }
  }
  const weightOffset = 2;
  const biasOffset = 1;
  const weight = new Float32Array(weightOffset + stableWidth);
  const bias = new Float32Array(biasOffset + stableWidth);
  for (let feature = 0; feature < stableWidth; ++feature) {
    weight[weightOffset + feature] = 0.75 + (feature % 11) * 0.025;
    bias[biasOffset + feature] = ((feature % 7) - 3) * 0.01;
  }

  const cases = [
    {
      name: "stable-affine-tail",
      rows: stableRows,
      width: stableWidth,
      inputOffset: stableInputOffset,
      outputOffset: 5,
      meanOffset: 2,
      rstdOffset: 1,
      weightOffset,
      biasOffset,
      hasWeight: true,
      hasBias: true,
      epsilon: 1e-5,
      dispatchX: 2,
      input: stableInput,
      weight,
      bias,
      tolerance: { output: 0.012, mean: 0.004, rstd: 0.004 },
    },
    {
      name: "no-affine-repeated-read-binding",
      rows: 3,
      width: 7,
      inputOffset: 1,
      outputOffset: 2,
      meanOffset: 1,
      rstdOffset: 3,
      weightOffset: 0,
      biasOffset: 0,
      hasWeight: false,
      hasBias: false,
      epsilon: 1e-12,
      dispatchX: 2,
      input: new Float32Array([
        -99, -3, -2, -1, 0, 1, 2, 3, 9, 4, -7, 2, 6, -5, 1, 8, 8, 7, 6, 5, 4, 3,
      ]),
      tolerance: { output: 2e-6, mean: 2e-6, rstd: 2e-6 },
    },
    {
      name: "bias-only",
      rows: 2,
      width: 5,
      inputOffset: 0,
      outputOffset: 0,
      meanOffset: 0,
      rstdOffset: 0,
      weightOffset: 0,
      biasOffset: 2,
      hasWeight: false,
      hasBias: true,
      epsilon: 1e-5,
      dispatchX: 2,
      input: new Float32Array([-2, 0, 1, 3, 7, 4, -1, 8, 2, 6]),
      bias: new Float32Array([99, 99, -0.2, -0.1, 0, 0.1, 0.2]),
      tolerance: { output: 2e-6, mean: 2e-6, rstd: 2e-6 },
    },
    {
      name: "weight-only",
      rows: 2,
      width: 5,
      inputOffset: 1,
      outputOffset: 1,
      meanOffset: 1,
      rstdOffset: 1,
      weightOffset: 1,
      biasOffset: 0,
      hasWeight: true,
      hasBias: false,
      epsilon: 1e-5,
      dispatchX: 1,
      input: new Float32Array([99, -2, 0, 1, 3, 7, 4, -1, 8, 2, 6]),
      weight: new Float32Array([99, 0.5, 0.75, 1, 1.25, 1.5]),
      tolerance: { output: 2e-6, mean: 2e-6, rstd: 2e-6 },
    },
  ];
  result.cases = {};
  for (const testCase of cases) {
    result.cases[testCase.name] = await execute(device, pipeline, testCase);
  }
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
  if (request.url === "/layer_norm.wgsl") {
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
  await page.waitForFunction(() => globalThis.layerNormTest?.done, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => globalThis.layerNormTest);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) throw new Error(result.message ?? result.errors.join("\n"));
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
