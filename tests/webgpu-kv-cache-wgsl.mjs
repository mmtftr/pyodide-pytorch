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
  "kv_cache_update.wgsl",
);

function existingFile(candidates) {
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return path.resolve(candidate);
  }
  return undefined;
}

const adapterMode = process.env.WEBGPU_ADAPTER ?? "swiftshader";
if (!new Set(["hardware", "swiftshader"]).has(adapterMode)) {
  throw new Error("unsupported WEBGPU_ADAPTER mode: " + adapterMode);
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
globalThis.kvCacheTest = result;

function gpuBuffer(device, label, values, usage) {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, values.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  if (values.byteLength !== 0) device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function paramsBuffer(device, testCase) {
  const bytes = new ArrayBuffer(128);
  const words = new Uint32Array(bytes);
  const elements =
    testCase.batch * testCase.heads * testCase.tokens * testCase.headDim;
  const workgroups = Math.ceil(elements / 64);
  const dispatchX = Math.min(workgroups, 65535);
  words.set([
    elements,
    testCase.batch,
    testCase.heads,
    testCase.tokens,
    testCase.headDim,
    testCase.capacity,
    testCase.keyStateOffset,
    testCase.valueStateOffset,
    testCase.keyCacheOffset,
    testCase.valueCacheOffset,
    testCase.positionOffset,
    testCase.positionWords,
    dispatchX,
    0,
    0,
    0,
    ...testCase.keyStateStrides,
    ...testCase.valueStateStrides,
    ...testCase.keyCacheStrides,
    ...testCase.valueCacheStrides,
  ]);
  return {
    buffer: gpuBuffer(
      device,
      testCase.name + " params",
      new Uint8Array(bytes),
      GPUBufferUsage.UNIFORM,
    ),
    dispatchX,
    dispatchY: Math.ceil(workgroups / dispatchX),
  };
}

function sourceIndex(offset, strides, batch, head, token, feature) {
  return offset + batch * strides[0] + head * strides[1] +
    token * strides[2] + feature * strides[3];
}

function makeCase(options) {
  const testCase = { ...options };
  const stateCoordinates = [];
  for (let batch = 0; batch < testCase.batch; ++batch) {
    for (let head = 0; head < testCase.heads; ++head) {
      for (let token = 0; token < testCase.tokens; ++token) {
        for (let feature = 0; feature < testCase.headDim; ++feature) {
          stateCoordinates.push([batch, head, token, feature]);
        }
      }
    }
  }
  const keyStateLength = Math.max(...stateCoordinates.map((coordinate) =>
    sourceIndex(
      testCase.keyStateOffset,
      testCase.keyStateStrides,
      ...coordinate,
    ),
  )) + 2;
  const valueStateLength = Math.max(...stateCoordinates.map((coordinate) =>
    sourceIndex(
      testCase.valueStateOffset,
      testCase.valueStateStrides,
      ...coordinate,
    ),
  )) + 2;
  testCase.keyStates = new Float32Array(keyStateLength).fill(-91.5);
  testCase.valueStates = new Float32Array(valueStateLength).fill(-92.5);
  for (const [batch, head, token, feature] of stateCoordinates) {
    const serial = batch * 1000 + head * 100 + token * 10 + feature;
    testCase.keyStates[sourceIndex(
      testCase.keyStateOffset,
      testCase.keyStateStrides,
      batch,
      head,
      token,
      feature,
    )] = 10000 + serial;
    testCase.valueStates[sourceIndex(
      testCase.valueStateOffset,
      testCase.valueStateStrides,
      batch,
      head,
      token,
      feature,
    )] = -10000 - serial;
  }

  const cacheElements = testCase.batch * testCase.heads *
    testCase.capacity * testCase.headDim;
  testCase.keyCache = new Float32Array(
    testCase.keyCacheOffset + cacheElements + 3,
  ).fill(-777.25);
  testCase.valueCache = new Float32Array(
    testCase.valueCacheOffset + cacheElements + 3,
  ).fill(-778.25);
  testCase.expectedKey = testCase.keyCache.slice();
  testCase.expectedValue = testCase.valueCache.slice();

  const positionElements = testCase.positionOffset + testCase.tokens + 2;
  testCase.positionData = new Uint32Array(
    positionElements * testCase.positionWords,
  );
  testCase.positionData.fill(0x7f7f7f7f);
  for (let token = 0; token < testCase.tokens; ++token) {
    const position = testCase.positions[token];
    const word = (testCase.positionOffset + token) * testCase.positionWords;
    testCase.positionData[word] = position >>> 0;
    if (testCase.positionWords === 2) {
      testCase.positionData[word + 1] = testCase.noncanonicalToken === token
        ? 0x12345678
        : (position < 0 ? 0xffffffff : 0);
    }
    if (
      position < 0 ||
      position >= testCase.capacity ||
      testCase.noncanonicalToken === token
    ) continue;
    for (let batch = 0; batch < testCase.batch; ++batch) {
      for (let head = 0; head < testCase.heads; ++head) {
        for (let feature = 0; feature < testCase.headDim; ++feature) {
          const coordinate = [batch, head, token, feature];
          const keyDestination = sourceIndex(
            testCase.keyCacheOffset,
            testCase.keyCacheStrides,
            batch,
            head,
            position,
            feature,
          );
          const valueDestination = sourceIndex(
            testCase.valueCacheOffset,
            testCase.valueCacheStrides,
            batch,
            head,
            position,
            feature,
          );
          testCase.expectedKey[keyDestination] = testCase.keyStates[
            sourceIndex(
              testCase.keyStateOffset,
              testCase.keyStateStrides,
              ...coordinate,
            )
          ];
          testCase.expectedValue[valueDestination] = testCase.valueStates[
            sourceIndex(
              testCase.valueStateOffset,
              testCase.valueStateStrides,
              ...coordinate,
            )
          ];
        }
      }
    }
  }
  return testCase;
}

function compareExact(actual, expected, label) {
  if (actual.length !== expected.length) throw new Error(label + " length");
  let changed = 0;
  for (let index = 0; index < expected.length; ++index) {
    if (!Object.is(actual[index], expected[index])) {
      throw new Error(
        label + " mismatch at " + index + ": " + actual[index] +
          " != " + expected[index],
      );
    }
    if (expected[index] !== expected[0]) changed += 1;
  }
  return changed;
}

async function execute(device, pipeline, testCase) {
  const readOnly = GPUBufferUsage.STORAGE;
  const writable = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const buffers = [
    gpuBuffer(device, testCase.name + " keys", testCase.keyStates, readOnly),
    gpuBuffer(device, testCase.name + " values", testCase.valueStates, readOnly),
    gpuBuffer(device, testCase.name + " positions", testCase.positionData, readOnly),
    gpuBuffer(device, testCase.name + " key cache", testCase.keyCache, writable),
    gpuBuffer(device, testCase.name + " value cache", testCase.valueCache, writable),
  ];
  const params = paramsBuffer(device, testCase);
  buffers.push(params.buffer);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    })),
  });
  const keyBytes = testCase.keyCache.byteLength;
  const valueBytes = testCase.valueCache.byteLength;
  const readback = device.createBuffer({
    label: testCase.name + " readback",
    size: keyBytes + valueBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(params.dispatchX, params.dispatchY, 1);
  pass.end();
  encoder.copyBufferToBuffer(buffers[3], 0, readback, 0, keyBytes);
  encoder.copyBufferToBuffer(buffers[4], 0, readback, keyBytes, valueBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0);
  readback.unmap();
  const actualKey = new Float32Array(bytes, 0, testCase.keyCache.length);
  const actualValue = new Float32Array(
    bytes,
    keyBytes,
    testCase.valueCache.length,
  );
  return {
    dispatches: 1,
    keyValuesChecked: actualKey.length,
    valueValuesChecked: actualValue.length,
    keyChanged: compareExact(actualKey, testCase.expectedKey, testCase.name + " key"),
    valueChanged: compareExact(
      actualValue,
      testCase.expectedValue,
      testCase.name + " value",
    ),
  };
}

try {
  const adapter = await navigator.gpu?.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const info = adapter.info ?? {};
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  device.addEventListener("uncapturederror", (event) => {
    result.errors.push(event.error?.message ?? String(event.error));
  });
  result.adapter = {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
  };

  const source = await (await fetch("/kv_cache_update.wgsl")).text();
  const module = device.createShaderModule({ code: source });
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
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const cases = [
    makeCase({
      name: "long-indexed-two-step-prefix",
      batch: 2,
      heads: 2,
      tokens: 2,
      headDim: 3,
      capacity: 6,
      positions: [1, 4],
      positionWords: 2,
      positionOffset: 1,
      keyStateOffset: 2,
      valueStateOffset: 3,
      keyCacheOffset: 2,
      valueCacheOffset: 3,
      keyStateStrides: [28, 12, 5, 1],
      valueStateStrides: [32, 14, 6, 1],
      keyCacheStrides: [36, 18, 3, 1],
      valueCacheStrides: [36, 18, 3, 1],
    }),
    makeCase({
      name: "int32-indexed-tail",
      batch: 1,
      heads: 1,
      tokens: 3,
      headDim: 2,
      capacity: 5,
      positions: [0, 3, 4],
      positionWords: 1,
      positionOffset: 2,
      keyStateOffset: 1,
      valueStateOffset: 2,
      keyCacheOffset: 1,
      valueCacheOffset: 2,
      keyStateStrides: [6, 6, 2, 1],
      valueStateStrides: [6, 6, 2, 1],
      keyCacheStrides: [10, 10, 2, 1],
      valueCacheStrides: [10, 10, 2, 1],
    }),
    makeCase({
      name: "invalid-and-noncanonical-long-positions-are-contained",
      batch: 1,
      heads: 1,
      tokens: 3,
      headDim: 2,
      capacity: 5,
      positions: [-1, 5, 2],
      noncanonicalToken: 2,
      positionWords: 2,
      positionOffset: 1,
      keyStateOffset: 0,
      valueStateOffset: 0,
      keyCacheOffset: 0,
      valueCacheOffset: 0,
      keyStateStrides: [6, 6, 2, 1],
      valueStateStrides: [6, 6, 2, 1],
      keyCacheStrides: [10, 10, 2, 1],
      valueCacheStrides: [10, 10, 2, 1],
    }),
  ];
  for (const testCase of cases) {
    result.cases[testCase.name] = await execute(device, pipeline, testCase);
  }
  const validationError = await device.popErrorScope();
  if (validationError) result.errors.push(validationError.message);
  if (result.errors.length !== 0) throw new Error(result.errors.join("\n"));
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
  if (pathname === "/kv_cache_update.wgsl") {
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
  const browserArguments = [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
    "--disable-dawn-features=use_dxc",
    "--enable-webgpu-developer-features",
    "--use-gpu-in-tests",
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
  await page.goto(
    "http://127.0.0.1:" + address.port + "/?adapter=" + adapterMode,
  );
  await page.waitForFunction(() => globalThis.kvCacheTest?.done, {
    timeout: 120_000,
  });
  const browserResult = await page.evaluate(() => globalThis.kvCacheTest);
  console.log(JSON.stringify(browserResult, null, 2));
  if (!browserResult.ok) {
    throw new Error(
      browserResult.message ?? browserResult.errors.join("\n"),
    );
  }
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
