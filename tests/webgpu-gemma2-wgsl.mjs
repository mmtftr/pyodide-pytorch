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
  "gt_tensor.wgsl",
  "triangular.wgsl",
  "mul_bool_inplace.wgsl",
  "where_float.wgsl",
  "gemma_rms_norm.wgsl",
];

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
for (const name of shaderNames) {
  const filename = path.join(shaderDirectory, name);
  if (!fs.existsSync(filename)) throw new Error("shader does not exist: " + filename);
}

const pageSource = String.raw`<!doctype html>
<meta charset="utf-8">
<script type="module">
const result = {
  done: false,
  ok: false,
  errors: [],
  requestedAdapter: new URLSearchParams(location.search).get("adapter"),
};
globalThis.gemma2ShaderTest = result;

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

function uniform(device, label, byteLength) {
  const bytes = new ArrayBuffer(byteLength);
  return {
    bytes,
    u32: new Uint32Array(bytes),
    i32: new Int32Array(bytes),
    f32: new Float32Array(bytes),
    finish() {
      return gpuBuffer(
        device,
        label,
        new Uint8Array(bytes),
        GPUBufferUsage.UNIFORM,
      );
    },
  };
}

async function pipeline(device, name) {
  const response = await fetch("/" + name);
  if (!response.ok) throw new Error(name + " fetch failed: " + response.status);
  const module = device.createShaderModule({ label: name, code: await response.text() });
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

function pass(encoder, pipelineValue, group, x = 1, y = 1) {
  const compute = encoder.beginComputePass();
  compute.setPipeline(pipelineValue);
  compute.setBindGroup(0, group);
  compute.dispatchWorkgroups(x, y);
  compute.end();
}

async function mappedCopy(device, source, byteLength) {
  const readback = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0);
  readback.unmap();
  readback.destroy();
  return bytes;
}

function longStorage(values, offset) {
  const words = new Uint32Array((offset + values.length + 1) * 2);
  words.fill(0x5a5a5a5a);
  values.forEach((value, index) => {
    const location = (offset + index) * 2;
    words[location] = value >>> 0;
    words[location + 1] = value < 0 ? 0xffffffff : 0;
  });
  return words;
}

function assertArray(actual, expected, label) {
  if (actual.length !== expected.length) {
    throw new Error(label + " length " + actual.length + " != " + expected.length);
  }
  for (let index = 0; index < expected.length; ++index) {
    if (
      actual[index] !== expected[index] &&
      !(Number.isNaN(actual[index]) && Number.isNaN(expected[index]))
    ) {
      throw new Error(
        label + " mismatch at " + index + ": " + actual[index] + " != " + expected[index],
      );
    }
  }
}

try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const device = await adapter.requestDevice();
  const adapterInfo = adapter.info ?? adapter.requestAdapterInfo?.();
  result.adapterInfo = adapterInfo
    ? {
        vendor: adapterInfo.vendor,
        architecture: adapterInfo.architecture,
        device: adapterInfo.device,
        description: adapterInfo.description,
      }
    : null;
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

  // Exact Gemma2 causal-mask core: arange(target) > cache_position[:, None],
  // float triu(diagonal=1), then in-place Float-by-Bool multiplication.
  const target = [0, 1, 2, 3, 4];
  const cachePosition = [0, 2, 4];
  const lhsOffset = 1;
  const rhsOffset = 2;
  const lhs = gpuBuffer(
    device,
    "gt Long lhs",
    longStorage(target, lhsOffset),
    GPUBufferUsage.STORAGE,
  );
  const rhs = gpuBuffer(
    device,
    "gt Long rhs",
    longStorage(cachePosition, rhsOffset),
    GPUBufferUsage.STORAGE,
  );
  const packedComparisonInitial = new Uint32Array(5).fill(0xabababab);
  const packedComparison = gpuBuffer(
    device,
    "gt packed Bool output",
    packedComparisonInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const gt = uniform(device, "gt params", 192);
  gt.u32.set([15, 2, 1, 2, lhsOffset, rhsOffset, 2, 1], 0);
  gt.u32.set([3, 5], 8);
  gt.u32.set([5], 16);
  gt.u32.set([1], 24);
  gt.u32.set([3, 1], 32);
  gt.u32.set([1, 1], 40);
  const gtParams = gt.finish();

  const minimum = -1000;
  const fullValues = new Float32Array(15).fill(minimum);
  const full = gpuBuffer(device, "full causal input", fullValues, GPUBufferUsage.STORAGE);
  const triangularInitial = new Float32Array(16).fill(12345.5);
  const causal = gpuBuffer(
    device,
    "causal output",
    triangularInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const triu = uniform(device, "triu params", 96);
  triu.u32.set([15, 2, 0, 0, 0, 0, 1], 0);
  triu.i32[7] = 1;
  triu.u32.set([3, 5], 8);
  triu.u32.set([5, 1], 16);
  const triuParams = triu.finish();

  const mul = uniform(device, "mul Bool params", 160);
  mul.u32.set([15, 2, 2, 0, 0, 1], 0);
  mul.u32.set([3, 5], 8);
  mul.u32.set([5, 1], 16);
  mul.u32.set([3, 5], 24);
  mul.u32.set([5, 1], 32);
  const mulParams = mul.finish();

  const causalEncoder = device.createCommandEncoder();
  pass(
    causalEncoder,
    pipelines["gt_tensor.wgsl"],
    bindGroup(device, pipelines["gt_tensor.wgsl"], [
      lhs,
      rhs,
      packedComparison,
      gtParams,
    ]),
  );
  pass(
    causalEncoder,
    pipelines["triangular.wgsl"],
    bindGroup(device, pipelines["triangular.wgsl"], [
      full,
      causal,
      triuParams,
    ]),
  );
  pass(
    causalEncoder,
    pipelines["mul_bool_inplace.wgsl"],
    bindGroup(device, pipelines["mul_bool_inplace.wgsl"], [
      causal,
      packedComparison,
      mulParams,
    ]),
  );
  device.queue.submit([causalEncoder.finish()]);

  const comparisonBytes = new Uint8Array(
    await mappedCopy(device, packedComparison, 20),
  );
  const expectedComparison = [];
  for (const position of cachePosition) {
    for (const key of target) expectedComparison.push(key > position ? 1 : 0);
  }
  assertArray(
    [...comparisonBytes.slice(0, 15)],
    expectedComparison,
    "broadcast restricted-Long gt.Tensor",
  );
  assertArray(
    [...comparisonBytes.slice(15, 16)],
    [0],
    "gt.Tensor canonical Bool tail",
  );
  assertArray(
    [...comparisonBytes.slice(16, 20)],
    [0xab, 0xab, 0xab, 0xab],
    "gt.Tensor output bound",
  );
  const causalValues = new Float32Array(await mappedCopy(device, causal, 64));
  const expectedCausal = [];
  for (let row = 0; row < 3; ++row) {
    for (let column = 0; column < 5; ++column) {
      expectedCausal.push(
        column - row >= 1 && column > cachePosition[row] ? minimum : 0,
      );
    }
  }
  assertArray(
    [...causalValues.slice(0, 15)],
    expectedCausal,
    "Gemma2 causal mask composition",
  );
  assertArray(
    [...causalValues.slice(15, 16)],
    [12345.5],
    "causal mask output bound",
  );

  // Sliding-window construction: odd-sized, offset packed-Bool input proves
  // that tril reads bytes correctly while owning each destination word.
  const boolInputBytes = new Uint8Array(20).fill(0xcc);
  boolInputBytes.fill(1, 1, 16);
  const boolInput = gpuBuffer(
    device,
    "offset packed Bool ones",
    boolInputBytes,
    GPUBufferUsage.STORAGE,
  );
  const slidingInitial = new Uint8Array(20).fill(0xdd);
  const sliding = gpuBuffer(
    device,
    "sliding Bool output",
    slidingInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const tril = uniform(device, "tril Bool params", 96);
  tril.u32.set([15, 4, 1, 0, 1, 1, 1], 0);
  tril.i32[7] = -2;
  tril.u32.set([1, 1, 3, 5], 8);
  tril.u32.set([15, 15, 5, 1], 16);
  const trilParams = tril.finish();
  const trilEncoder = device.createCommandEncoder();
  pass(
    trilEncoder,
    pipelines["triangular.wgsl"],
    bindGroup(device, pipelines["triangular.wgsl"], [
      boolInput,
      sliding,
      trilParams,
    ]),
  );
  device.queue.submit([trilEncoder.finish()]);
  const slidingBytes = new Uint8Array(await mappedCopy(device, sliding, 20));
  const expectedSliding = [];
  for (let row = 0; row < 3; ++row) {
    for (let column = 0; column < 5; ++column) {
      expectedSliding.push(column - row <= -2 ? 1 : 0);
    }
  }
  assertArray(
    [...slidingBytes.slice(0, 15)],
    expectedSliding,
    "packed Bool tril",
  );
  assertArray([...slidingBytes.slice(15, 16)], [0], "tril canonical Bool tail");
  assertArray(
    [...slidingBytes.slice(16, 20)],
    [0xdd, 0xdd, 0xdd, 0xdd],
    "tril output bound",
  );

  // where.self accepts the exact scalar/tensor branch emitted by
  // torch.where(sliding_window_mask, min_dtype, attention_mask).
  const attentionStorage = new Float32Array(17).fill(7777);
  const attentionOffset = 1;
  for (let index = 0; index < 15; ++index) {
    attentionStorage[attentionOffset + index] = index + 0.25;
  }
  const attention = gpuBuffer(
    device,
    "where attention",
    attentionStorage,
    GPUBufferUsage.STORAGE,
  );
  const whereInitial = new Float32Array(16).fill(-4321);
  const selected = gpuBuffer(
    device,
    "where output",
    whereInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const where = uniform(device, "where params", 288);
  where.u32.set([15, 4, 4, 0, 4, 0, 0, attentionOffset, 1, 0], 0);
  where.f32[10] = minimum;
  where.u32[12] = 0;
  where.u32[13] = 1;
  where.u32.set([1, 1, 3, 5], 16);
  where.u32.set([1, 1, 3, 5], 24);
  where.u32.set([15, 15, 5, 1], 32);
  where.u32.set([1, 1, 3, 5], 56);
  where.u32.set([15, 15, 5, 1], 64);
  const whereParams = where.finish();
  const whereEncoder = device.createCommandEncoder();
  pass(
    whereEncoder,
    pipelines["where_float.wgsl"],
    bindGroup(device, pipelines["where_float.wgsl"], [
      sliding,
      sliding,
      attention,
      selected,
      whereParams,
    ]),
  );
  device.queue.submit([whereEncoder.finish()]);
  const selectedValues = new Float32Array(await mappedCopy(device, selected, 64));
  const expectedSelected = expectedSliding.map((masked, index) =>
    masked ? minimum : index + 0.25,
  );
  assertArray(
    [...selectedValues.slice(0, 15)],
    expectedSelected,
    "where.self scalar/tensor selection",
  );
  assertArray(
    [...selectedValues.slice(15, 16)],
    [-4321],
    "where.self output bound",
  );

  // Gemma offset-weight RMSNorm uses a reduction tail wider than one workgroup.
  const rows = 2;
  const width = 518;
  const inputOffset = 3;
  const weightOffset = 2;
  const normInput = new Float32Array(inputOffset + rows * width + 2);
  const normWeight = new Float32Array(weightOffset + width + 2);
  for (let index = 0; index < rows * width; ++index) {
    normInput[inputOffset + index] =
      0.6 * Math.sin(index * 0.071) + 0.2 * Math.cos(index * 0.019);
  }
  for (let index = 0; index < width; ++index) {
    normWeight[weightOffset + index] = 0.12 * Math.sin(index * 0.037);
  }
  const normInputBuffer = gpuBuffer(
    device,
    "Gemma norm input",
    normInput,
    GPUBufferUsage.STORAGE,
  );
  const normWeightBuffer = gpuBuffer(
    device,
    "Gemma offset weight",
    normWeight,
    GPUBufferUsage.STORAGE,
  );
  const normOutput = gpuBuffer(
    device,
    "Gemma norm output",
    new Float32Array(rows * width),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const norm = uniform(device, "Gemma norm params", 32);
  norm.u32.set([rows, width, inputOffset, 0, weightOffset, rows], 0);
  norm.f32[6] = 1e-6;
  const normParams = norm.finish();
  const normEncoder = device.createCommandEncoder();
  pass(
    normEncoder,
    pipelines["gemma_rms_norm.wgsl"],
    bindGroup(device, pipelines["gemma_rms_norm.wgsl"], [
      normInputBuffer,
      normWeightBuffer,
      normOutput,
      normParams,
    ]),
    rows,
  );
  device.queue.submit([normEncoder.finish()]);
  const normActual = new Float32Array(
    await mappedCopy(device, normOutput, rows * width * 4),
  );
  let maxNormError = 0;
  for (let row = 0; row < rows; ++row) {
    let squareSum = 0;
    for (let feature = 0; feature < width; ++feature) {
      const value = normInput[inputOffset + row * width + feature];
      squareSum += value * value;
    }
    const inverse = 1 / Math.sqrt(squareSum / width + 1e-6);
    for (let feature = 0; feature < width; ++feature) {
      const index = row * width + feature;
      const expected =
        normInput[inputOffset + index] * inverse *
        (1 + normWeight[weightOffset + feature]);
      const error = Math.abs(normActual[index] - expected);
      maxNormError = Math.max(maxNormError, error);
      const allowed = 3e-5 + 3e-5 * Math.abs(expected);
      if (!Number.isFinite(normActual[index]) || error > allowed) {
        throw new Error(
          "Gemma RMSNorm mismatch at " + index + ": actual=" +
            normActual[index] + " expected=" + expected + " error=" + error,
        );
      }
    }
  }

  const validationError = await device.popErrorScope();
  if (validationError) result.errors.push(validationError.message);
  if (result.errors.length !== 0) throw new Error(result.errors.join("\n"));
  result.compiled = ${JSON.stringify(shaderNames)};
  result.causalMask = expectedCausal;
  result.slidingMask = expectedSliding;
  result.maxNormError = maxNormError;
  result.dispatches = {
    gtTensor: 1,
    triu: 1,
    mulBoolInplace: 1,
    trilBool: 1,
    whereSelf: 1,
    gemmaRmsNorm: 1,
  };
  result.ok = true;
} catch (error) {
  result.message = error instanceof Error ? error.stack : String(error);
} finally {
  result.done = true;
}
</script>`;

const server = http.createServer((request, response) => {
  const name = request.url.slice(1).split("?", 1)[0];
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
  page.on("console", (message) => console.log("Gemma2 shader: " + message.text()));
  page.on("pageerror", (error) => console.error(error.stack ?? String(error)));
  await page.goto(
    `http://127.0.0.1:${address.port}/?adapter=${encodeURIComponent(adapterMode)}`,
  );
  await page.waitForFunction(() => globalThis.gemma2ShaderTest?.done, {
    timeout: 120_000,
  });
  const result = await page.evaluate(() => globalThis.gemma2ShaderTest);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) throw new Error(result.message ?? "Gemma2 shader test failed");
} catch (error) {
  console.error(
    JSON.stringify(
      {
        gemma2ShaderTest: "failed",
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
