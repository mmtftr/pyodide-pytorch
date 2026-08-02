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
const shaders = new Map([
  ["/swiglu_gemv.wgsl", path.join(shaderDirectory, "swiglu_gemv.wgsl")],
  [
    "/swiglu_gemv_subgroup_s4.wgsl",
    path.join(shaderDirectory, "swiglu_gemv_subgroup_s4.wgsl"),
  ],
]);

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
for (const shader of shaders.values()) {
  if (!fs.existsSync(shader)) throw new Error("shader does not exist: " + shader);
}

const pageSource = String.raw`<!doctype html>
<meta charset="utf-8">
<script type="module">
const result = {
  done: false,
  ok: false,
  errors: [],
  requestedAdapter: new URLSearchParams(location.search).get("adapter"),
  variants: {},
};
globalThis.swiGluTest = result;

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
  const bytes = new ArrayBuffer(64);
  const words = new Uint32Array(bytes);
  words[0] = testCase.columns;
  words[1] = testCase.inner;
  words[2] = testCase.inputOffset;
  words[3] = testCase.gateWeightOffset;
  words[4] = testCase.upWeightOffset;
  words[5] = testCase.gateBiasOffset;
  words[6] = testCase.upBiasOffset;
  words[7] = testCase.outputOffset;
  words[8] = testCase.hasGateBias ? 1 : 0;
  words[9] = testCase.hasUpBias ? 1 : 0;
  return gpuBuffer(
    device,
    testCase.name + " params",
    new Uint8Array(bytes),
    GPUBufferUsage.UNIFORM,
  );
}

function patterned(length, scale, phase) {
  const values = new Float32Array(length);
  for (let index = 0; index < length; ++index) {
    values[index] =
      (Math.sin((index + phase) * 0.37) +
        0.5 * Math.cos((index + phase) * 0.11)) *
      scale;
  }
  return values;
}

function makeCase(options) {
  const testCase = { ...options };
  testCase.input = new Float32Array(testCase.inputOffset + testCase.inner + 2);
  testCase.gateWeight = new Float32Array(
    testCase.gateWeightOffset + testCase.columns * testCase.inner + 2,
  );
  testCase.upWeight = new Float32Array(
    testCase.upWeightOffset + testCase.columns * testCase.inner + 2,
  );
  testCase.gateBias = new Float32Array(
    testCase.gateBiasOffset + testCase.columns + 2,
  );
  testCase.upBias = new Float32Array(
    testCase.upBiasOffset + testCase.columns + 2,
  );
  testCase.input.set(
    patterned(testCase.inner, 0.45, 1),
    testCase.inputOffset,
  );
  testCase.gateWeight.set(
    patterned(testCase.columns * testCase.inner, 0.035, 7),
    testCase.gateWeightOffset,
  );
  testCase.upWeight.set(
    patterned(testCase.columns * testCase.inner, 0.028, 19),
    testCase.upWeightOffset,
  );
  if (testCase.hasGateBias) {
    testCase.gateBias.set(
      patterned(testCase.columns, 0.08, 29),
      testCase.gateBiasOffset,
    );
  }
  if (testCase.hasUpBias) {
    testCase.upBias.set(
      patterned(testCase.columns, 0.06, 41),
      testCase.upBiasOffset,
    );
  }
  return testCase;
}

function reference(testCase) {
  const expected = new Float32Array(testCase.columns);
  for (let column = 0; column < testCase.columns; ++column) {
    let gate = 0;
    let up = 0;
    const gateBase =
      testCase.gateWeightOffset + column * testCase.inner;
    const upBase = testCase.upWeightOffset + column * testCase.inner;
    for (let feature = 0; feature < testCase.inner; ++feature) {
      const inputValue =
        testCase.input[testCase.inputOffset + feature];
      gate += inputValue * testCase.gateWeight[gateBase + feature];
      up += inputValue * testCase.upWeight[upBase + feature];
    }
    if (testCase.hasGateBias) {
      gate += testCase.gateBias[testCase.gateBiasOffset + column];
    }
    if (testCase.hasUpBias) {
      up += testCase.upBias[testCase.upBiasOffset + column];
    }
    expected[column] = (gate / (1 + Math.exp(-gate))) * up;
  }
  return expected;
}

function compare(actual, expected, testCase, variant) {
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  for (let index = 0; index < expected.length; ++index) {
    const absolute = Math.abs(actual[index] - expected[index]);
    const relative = absolute / Math.max(Math.abs(expected[index]), 1e-6);
    maxAbsoluteError = Math.max(maxAbsoluteError, absolute);
    maxRelativeError = Math.max(maxRelativeError, relative);
    const allowed = 8e-5 + 8e-4 * Math.abs(expected[index]);
    if (!Number.isFinite(actual[index]) || absolute > allowed) {
      throw new Error(
        testCase.name + " " + variant + " mismatch at " + index +
          ": actual=" + actual[index] + " expected=" + expected[index] +
          " error=" + absolute + " allowed=" + allowed,
      );
    }
  }
  return { maxAbsoluteError, maxRelativeError };
}

async function compilePipeline(device, route) {
  const source = await (await fetch(route)).text();
  const module = device.createShaderModule({ code: source });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(
    (message) => message.type === "error",
  );
  if (errors.length !== 0) {
    throw new Error(
      route + ": " + errors.map((message) => message.message).join("\n"),
    );
  }
  return device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
}

async function execute(device, pipeline, testCase, variant) {
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const input = gpuBuffer(
    device,
    testCase.name + " input",
    testCase.input,
    storage,
  );
  const gateWeight = gpuBuffer(
    device,
    testCase.name + " gate weight",
    testCase.gateWeight,
    storage,
  );
  const upWeight = gpuBuffer(
    device,
    testCase.name + " up weight",
    testCase.upWeight,
    storage,
  );
  const gateBias = testCase.hasGateBias
    ? gpuBuffer(
        device,
        testCase.name + " gate bias",
        testCase.gateBias,
        storage,
      )
    : input;
  const upBias = testCase.hasUpBias
    ? gpuBuffer(
        device,
        testCase.name + " up bias",
        testCase.upBias,
        storage,
      )
    : input;
  const outputValues = new Float32Array(
    testCase.outputOffset + testCase.columns + 3,
  );
  outputValues.fill(-777.25);
  const output = gpuBuffer(
    device,
    testCase.name + " output",
    outputValues,
    storage,
  );
  const params = paramsBuffer(device, testCase);
  const buffers = [
    input,
    gateWeight,
    upWeight,
    gateBias,
    upBias,
    output,
    params,
  ];
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    })),
  });
  const readback = device.createBuffer({
    label: testCase.name + " readback",
    size: outputValues.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const outputsPerWorkgroup = variant === "subgroup-s4" ? 4 : 64;
  pass.dispatchWorkgroups(
    Math.ceil(testCase.columns / outputsPerWorkgroup),
    1,
    1,
  );
  pass.end();
  encoder.copyBufferToBuffer(
    output,
    0,
    readback,
    0,
    outputValues.byteLength,
  );
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const allOutput = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  for (let index = 0; index < testCase.outputOffset; ++index) {
    if (allOutput[index] !== -777.25) {
      throw new Error(testCase.name + " overwrote output prefix sentinel");
    }
  }
  for (
    let index = testCase.outputOffset + testCase.columns;
    index < allOutput.length;
    ++index
  ) {
    if (allOutput[index] !== -777.25) {
      throw new Error(testCase.name + " overwrote output suffix sentinel");
    }
  }
  const actual = allOutput.subarray(
    testCase.outputOffset,
    testCase.outputOffset + testCase.columns,
  );
  return {
    ...compare(actual, reference(testCase), testCase, variant),
    dispatches: 1,
  };
}

try {
  const adapter = await navigator.gpu?.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const info = adapter.info ?? {};
  const fixed32Subgroups =
    adapter.features.has("subgroups") &&
    info.subgroupMinSize === 32 &&
    info.subgroupMaxSize === 32;
  const descriptor = fixed32Subgroups
    ? { requiredFeatures: ["subgroups"] }
    : {};
  const device = await adapter.requestDevice(descriptor);
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
  result.subgroups = {
    feature: adapter.features.has("subgroups"),
    minSize: info.subgroupMinSize ?? null,
    maxSize: info.subgroupMaxSize ?? null,
    fixed32: fixed32Subgroups,
  };

  const cases = [
    makeCase({
      name: "decode-tail-no-bias-repeated-read-binding",
      inner: 259,
      columns: 71,
      inputOffset: 3,
      gateWeightOffset: 5,
      upWeightOffset: 7,
      gateBiasOffset: 0,
      upBiasOffset: 0,
      outputOffset: 4,
      hasGateBias: false,
      hasUpBias: false,
    }),
    makeCase({
      name: "independent-biases-and-storage-offsets",
      inner: 37,
      columns: 9,
      inputOffset: 2,
      gateWeightOffset: 11,
      upWeightOffset: 13,
      gateBiasOffset: 2,
      upBiasOffset: 4,
      outputOffset: 3,
      hasGateBias: true,
      hasUpBias: true,
    }),
    makeCase({
      name: "gate-bias-only-small-tail",
      inner: 5,
      columns: 5,
      inputOffset: 1,
      gateWeightOffset: 3,
      upWeightOffset: 4,
      gateBiasOffset: 3,
      upBiasOffset: 0,
      outputOffset: 2,
      hasGateBias: true,
      hasUpBias: false,
    }),
  ];

  const portable = await compilePipeline(device, "/swiglu_gemv.wgsl");
  result.variants.portable = { cases: {}, dispatches: 0 };
  for (const testCase of cases) {
    const evidence = await execute(device, portable, testCase, "portable");
    result.variants.portable.cases[testCase.name] = evidence;
    result.variants.portable.dispatches += evidence.dispatches;
  }
  if (fixed32Subgroups) {
    const subgroup = await compilePipeline(
      device,
      "/swiglu_gemv_subgroup_s4.wgsl",
    );
    result.variants["subgroup-s4"] = { cases: {}, dispatches: 0 };
    for (const testCase of cases) {
      const evidence = await execute(
        device,
        subgroup,
        testCase,
        "subgroup-s4",
      );
      result.variants["subgroup-s4"].cases[testCase.name] = evidence;
      result.variants["subgroup-s4"].dispatches += evidence.dispatches;
    }
  } else {
    result.variants["subgroup-s4"] = {
      skipped: "adapter does not expose fixed-size-32 subgroups",
    };
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
  const pathname = new URL(request.url, "http://127.0.0.1").pathname;
  const shader = shaders.get(pathname);
  if (shader) {
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
  await page.goto(
    "http://127.0.0.1:" + address.port + "/?adapter=" + adapterMode,
  );
  await page.waitForFunction(() => globalThis.swiGluTest?.done, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => globalThis.swiGluTest);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) {
    throw new Error(result.message ?? result.errors.join("\n"));
  }
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
