import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const shaderName = "masked_fill_scalar.wgsl";
const shader = path.resolve(
  scriptDirectory,
  "..",
  "webgpu",
  "llm_kernels",
  shaderName,
);

function existingFile(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return path.resolve(candidate);
  }
  return undefined;
}

const adapterMode = process.env.WEBGPU_ADAPTER ?? "swiftshader";
if (!new Set(["hardware", "swiftshader"]).has(adapterMode)) {
  throw new Error("unsupported WebGPU adapter mode: " + adapterMode);
}
const chrome = existingFile([
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
]);
if (!chrome) throw new Error("Chrome executable was not found");
if (!fs.existsSync(shader)) throw new Error("shader does not exist: " + shader);

const pageSource = String.raw`<!doctype html>
<meta charset="utf-8">
<script type="module">
const result = {
  done: false,
  ok: false,
  errors: [],
  requestedAdapter: new URLSearchParams(location.search).get("adapter"),
  cases: {},
};
globalThis.maskedFillShaderTest = result;

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

function product(sizes) {
  return sizes.reduce((value, size) => value * size, 1);
}

function storageIndex(linearIndex, sizes, strides, offset) {
  let remaining = linearIndex;
  let index = offset;
  for (let dim = sizes.length - 1; dim >= 0; --dim) {
    const coordinate = remaining % sizes[dim];
    remaining = Math.floor(remaining / sizes[dim]);
    index += coordinate * strides[dim];
  }
  return index;
}

function broadcastIndex(linearIndex, outputSizes, inputSizes, inputStrides, offset) {
  let remaining = linearIndex;
  let index = offset;
  for (let reverseDim = 0; reverseDim < outputSizes.length; ++reverseDim) {
    const dim = outputSizes.length - reverseDim - 1;
    const coordinate = remaining % outputSizes[dim];
    remaining = Math.floor(remaining / outputSizes[dim]);
    if (dim + inputSizes.length >= outputSizes.length) {
      const inputDim = dim + inputSizes.length - outputSizes.length;
      if (inputSizes[inputDim] !== 1) {
        index += coordinate * inputStrides[inputDim];
      }
    }
  }
  return index;
}

function maximumStorageIndex(sizes, strides, offset) {
  let maximum = offset;
  for (let dim = 0; dim < sizes.length; ++dim) {
    maximum += (sizes[dim] - 1) * strides[dim];
  }
  return maximum;
}

function floatBits(value) {
  const bytes = new ArrayBuffer(4);
  new Float32Array(bytes)[0] = value;
  return new Uint32Array(bytes)[0];
}

function splitMetadata(words, firstWord, values) {
  for (let index = 0; index < values.length; ++index) {
    words[firstWord + index] = values[index];
  }
}

function assertFloat(actual, expected, label) {
  if (Number.isNaN(expected)) {
    if (!Number.isNaN(actual)) throw new Error(label + " expected NaN, got " + actual);
    return;
  }
  if (!Object.is(actual, expected) && Math.abs(actual - expected) > 1e-6) {
    throw new Error(label + ": " + actual + " != " + expected);
  }
}

async function mappedFloats(device, source, length) {
  const byteLength = length * 4;
  const readback = device.createBuffer({
    label: "masked_fill readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return values;
}

async function runCase(device, pipeline, testCase) {
  const outputLength = product(testCase.outputSizes);
  const selfLength = maximumStorageIndex(
    testCase.selfSizes,
    testCase.selfStrides,
    testCase.selfOffset,
  ) + 4;
  const maskLength = maximumStorageIndex(
    testCase.maskSizes,
    testCase.maskStrides,
    testCase.maskOffset,
  ) + 4;
  const selfStorage = new Float32Array(selfLength).fill(-9876.5);
  const maskStorage = new Uint8Array((maskLength + 3) & ~3).fill(0x7e);

  const selfElements = product(testCase.selfSizes);
  for (let linear = 0; linear < selfElements; ++linear) {
    selfStorage[storageIndex(
      linear,
      testCase.selfSizes,
      testCase.selfStrides,
      testCase.selfOffset,
    )] = testCase.selfValue(linear);
  }
  const maskElements = product(testCase.maskSizes);
  for (let linear = 0; linear < maskElements; ++linear) {
    maskStorage[storageIndex(
      linear,
      testCase.maskSizes,
      testCase.maskStrides,
      testCase.maskOffset,
    )] = testCase.maskValue(linear);
  }

  const guard = 7777.25;
  const outputStorage = new Float32Array(
    testCase.outputOffset + outputLength + 4,
  ).fill(guard);
  const selfBuffer = gpuBuffer(
    device,
    testCase.name + " self",
    selfStorage,
    GPUBufferUsage.STORAGE,
  );
  const maskBuffer = gpuBuffer(
    device,
    testCase.name + " packed Bool mask",
    maskStorage,
    GPUBufferUsage.STORAGE,
  );
  const outputBuffer = gpuBuffer(
    device,
    testCase.name + " output",
    outputStorage,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );

  const workgroups = Math.ceil(outputLength / 64);
  const dispatchX = Math.min(workgroups, 65535);
  const dispatchY = Math.ceil(workgroups / dispatchX);
  const paramsBytes = new ArrayBuffer(208);
  const params = new Uint32Array(paramsBytes);
  params.set([
    outputLength,
    testCase.outputSizes.length,
    testCase.selfSizes.length,
    testCase.maskSizes.length,
    testCase.selfOffset,
    testCase.maskOffset,
    testCase.outputOffset,
    floatBits(testCase.fillValue),
    dispatchX,
  ]);
  splitMetadata(params, 12, testCase.outputSizes);
  splitMetadata(params, 20, testCase.selfSizes);
  splitMetadata(params, 28, testCase.selfStrides);
  splitMetadata(params, 36, testCase.maskSizes);
  splitMetadata(params, 44, testCase.maskStrides);
  const paramsBuffer = gpuBuffer(
    device,
    testCase.name + " params",
    new Uint8Array(paramsBytes),
    GPUBufferUsage.UNIFORM,
  );
  const bindGroup = device.createBindGroup({
    label: testCase.name + " bind group",
    layout: pipeline.getBindGroupLayout(0),
    entries: [selfBuffer, maskBuffer, outputBuffer, paramsBuffer].map(
      (buffer, binding) => ({ binding, resource: { buffer } }),
    ),
  });

  const encoder = device.createCommandEncoder({ label: testCase.name });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dispatchX, dispatchY);
  pass.end();
  device.queue.submit([encoder.finish()]);

  const actual = await mappedFloats(device, outputBuffer, outputStorage.length);
  for (let index = 0; index < testCase.outputOffset; ++index) {
    assertFloat(actual[index], guard, testCase.name + " prefix guard " + index);
  }
  let masked = 0;
  for (let linear = 0; linear < outputLength; ++linear) {
    const selfIndex = broadcastIndex(
      linear,
      testCase.outputSizes,
      testCase.selfSizes,
      testCase.selfStrides,
      testCase.selfOffset,
    );
    const maskIndex = broadcastIndex(
      linear,
      testCase.outputSizes,
      testCase.maskSizes,
      testCase.maskStrides,
      testCase.maskOffset,
    );
    const isMasked = maskStorage[maskIndex] !== 0;
    const expected = isMasked
      ? Math.fround(testCase.fillValue)
      : selfStorage[selfIndex];
    assertFloat(
      actual[testCase.outputOffset + linear],
      expected,
      testCase.name + " output " + linear,
    );
    if (isMasked) masked += 1;
  }
  for (
    let index = testCase.outputOffset + outputLength;
    index < actual.length;
    ++index
  ) {
    assertFloat(actual[index], guard, testCase.name + " suffix guard " + index);
  }
  for (const buffer of [selfBuffer, maskBuffer, outputBuffer, paramsBuffer]) {
    buffer.destroy();
  }
  return {
    outputShape: testCase.outputSizes,
    valuesChecked: outputLength,
    masked,
    unmasked: outputLength - masked,
    dispatches: 1,
  };
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const adapterInfo = adapter.info ?? {};
  result.adapterInfo = {
    vendor: adapterInfo.vendor ?? "",
    architecture: adapterInfo.architecture ?? "",
    device: adapterInfo.device ?? "",
    description: adapterInfo.description ?? "",
  };
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  device.addEventListener("uncapturederror", (event) => {
    result.errors.push(event.error?.message ?? String(event.error));
  });

  const source = await (await fetch("/${shaderName}")).text();
  const module = device.createShaderModule({
    label: "masked_fill.Scalar",
    code: source,
  });
  const compilation = await module.getCompilationInfo();
  const compilationErrors = compilation.messages.filter(
    (message) => message.type === "error",
  );
  if (compilationErrors.length !== 0) {
    throw new Error(
      compilationErrors.map((message) => message.message).join("\n"),
    );
  }
  const pipeline = await device.createComputePipelineAsync({
    label: "masked_fill.Scalar",
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const cases = [
    {
      name: "attention rank4 bidirectional broadcast negative infinity tail",
      outputSizes: [2, 3, 5, 7],
      selfSizes: [1, 3, 1, 7],
      selfStrides: [101, 23, 15, 2],
      selfOffset: 3,
      maskSizes: [2, 1, 5, 1],
      maskStrides: [37, 31, 6, 2],
      maskOffset: 1,
      outputOffset: 5,
      fillValue: -Infinity,
      selfValue: (linear) => Math.fround(linear * 0.25 - 4),
      maskValue: (linear) => linear % 4 === 0 ? 255 : (linear % 7 === 0 ? 2 : 0),
    },
    {
      name: "rank8 scalar self positive infinity",
      outputSizes: [2, 1, 2, 1, 2, 1, 2, 5],
      selfSizes: [],
      selfStrides: [],
      selfOffset: 2,
      maskSizes: [2, 1, 2, 1, 2, 1, 2, 5],
      maskStrides: [95, 91, 47, 43, 23, 19, 11, 2],
      maskOffset: 3,
      outputOffset: 1,
      fillValue: Infinity,
      selfValue: () => 6.75,
      maskValue: (linear) => linear % 5 === 1 ? 1 : 0,
    },
    {
      name: "finite fill rank1 mask broadcast and noncanonical true bytes",
      outputSizes: [3, 5],
      selfSizes: [3, 5],
      selfStrides: [9, 1],
      selfOffset: 4,
      maskSizes: [5],
      maskStrides: [2],
      maskOffset: 3,
      outputOffset: 3,
      fillValue: -3.25,
      selfValue: (linear) => Math.fround(linear * 1.125 + 0.5),
      maskValue: (linear) => linear === 1 ? 2 : (linear === 4 ? 255 : 0),
    },
    {
      name: "scalar Bool mask NaN scalar",
      outputSizes: [9],
      selfSizes: [9],
      selfStrides: [2],
      selfOffset: 1,
      maskSizes: [],
      maskStrides: [],
      maskOffset: 3,
      outputOffset: 2,
      fillValue: NaN,
      selfValue: (linear) => Math.fround(linear - 4.5),
      maskValue: () => 255,
    },
  ];
  for (const testCase of cases) {
    result.cases[testCase.name] = await runCase(device, pipeline, testCase);
  }

  const validationError = await device.popErrorScope();
  if (validationError) result.errors.push(validationError.message);
  if (result.errors.length !== 0) throw new Error(result.errors.join("\n"));
  result.compiled = [${JSON.stringify(shaderName)}];
  result.dispatches = cases.length;
  result.ok = true;
} catch (error) {
  result.message = error instanceof Error ? error.stack : String(error);
} finally {
  result.done = true;
}
</script>`;

const server = http.createServer((request, response) => {
  const pathname = new URL(request.url, "http://127.0.0.1").pathname;
  if (pathname === "/" + shaderName) {
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
  const args = [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
    "--use-gpu-in-tests",
  ];
  if (adapterMode === "swiftshader") {
    args.push("--enable-unsafe-swiftshader", "--use-webgpu-adapter=swiftshader");
  }
  browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args,
  });
  const page = await browser.newPage();
  page.on("pageerror", (error) => console.error(error.stack ?? String(error)));
  await page.goto(
    "http://127.0.0.1:" + address.port + "/?adapter=" + adapterMode,
  );
  await page.waitForFunction(() => globalThis.maskedFillShaderTest?.done, {
    timeout: 120_000,
  });
  const browserResult = await page.evaluate(
    () => globalThis.maskedFillShaderTest,
  );
  console.log(JSON.stringify(browserResult, null, 2));
  if (!browserResult.ok) {
    throw new Error(
      browserResult.message ?? browserResult.errors.join("\n"),
    );
  }
} catch (error) {
  console.error(
    JSON.stringify(
      {
        maskedFillShaderTest: "failed",
        adapterMode,
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
