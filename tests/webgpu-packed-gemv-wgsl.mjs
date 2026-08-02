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
  [
    "/linear_gemv_q8_s4.wgsl",
    path.join(shaderDirectory, "linear_gemv_q8_s4.wgsl"),
  ],
  [
    "/linear_gemv_subgroup_s4.wgsl",
    path.join(shaderDirectory, "linear_gemv_subgroup_s4.wgsl"),
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
const extended = process.env.PACKED_GEMV_EXTENDED === "1";
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
const GROUP_SIZE = 128;
const VALUES_PER_WORD = 4;
const OUTPUTS_PER_SUBGROUP = 4;
const FORMAT_VERSION = 1;
const params = new URLSearchParams(location.search);
const result = {
  done: false,
  ok: false,
  errors: [],
  requestedAdapter: params.get("adapter"),
  extended: params.get("extended") === "1",
};
globalThis.packedGemvTest = result;

function gpuBuffer(device, label, values, usage) {
  const buffer = device.createBuffer({
    label,
    size: Math.max(4, values.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  if (values.byteLength !== 0) device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function patterned(length, scale, seed) {
  const values = new Float32Array(length);
  let state = seed >>> 0;
  for (let index = 0; index < length; ++index) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values[index] = (((state >>> 8) / 0x01000000) * 2 - 1) * scale;
  }
  return values;
}

function makeCase(options) {
  const testCase = {
    inputOffset: 0,
    weightOffset: 0,
    packedWeightOffset: 0,
    scaleOffset: 0,
    biasOffset: 0,
    outputOffset: 0,
    hasBias: false,
    rows: 1,
    groupSize: GROUP_SIZE,
    formatVersion: FORMAT_VERSION,
    ...options,
  };
  testCase.input = new Float32Array(
    testCase.inputOffset + testCase.inner + 2,
  );
  testCase.input.set(
    patterned(testCase.inner, 0.55, testCase.seed + 1),
    testCase.inputOffset,
  );
  testCase.weight = new Float32Array(
    testCase.weightOffset + testCase.columns * testCase.inner + 2,
  );
  testCase.weight.set(
    patterned(
      testCase.columns * testCase.inner,
      0.025,
      testCase.seed + 7,
    ),
    testCase.weightOffset,
  );
  testCase.bias = new Float32Array(
    testCase.biasOffset + testCase.columns + 2,
  );
  if (testCase.hasBias) {
    testCase.bias.set(
      patterned(testCase.columns, 0.015, testCase.seed + 19),
      testCase.biasOffset,
    );
  }
  return testCase;
}

function packQ8(testCase) {
  if (testCase.inner <= 0 || testCase.inner % GROUP_SIZE !== 0) {
    throw new Error("Q8 format requires a positive K divisible by 128");
  }
  const wordsPerRow = testCase.inner / VALUES_PER_WORD;
  const groupsPerRow = testCase.inner / GROUP_SIZE;
  const words = new Uint32Array(
    testCase.packedWeightOffset + testCase.columns * wordsPerRow + 2,
  );
  words.fill(0xdeadbeef);
  const scales = new Float32Array(
    testCase.scaleOffset + testCase.columns * groupsPerRow + 2,
  );
  scales.fill(Number.NaN);

  for (let row = 0; row < testCase.columns; ++row) {
    const sourceRow = testCase.weightOffset + row * testCase.inner;
    const packedRow = testCase.packedWeightOffset + row * wordsPerRow;
    const scaleRow = testCase.scaleOffset + row * groupsPerRow;
    for (let group = 0; group < groupsPerRow; ++group) {
      const sourceGroup = sourceRow + group * GROUP_SIZE;
      let absmax = 0;
      for (let index = 0; index < GROUP_SIZE; ++index) {
        absmax = Math.max(absmax, Math.abs(testCase.weight[sourceGroup + index]));
      }
      scales[scaleRow + group] = absmax;
      for (let word = 0; word < GROUP_SIZE / VALUES_PER_WORD; ++word) {
        let packed = 0;
        for (let byte = 0; byte < VALUES_PER_WORD; ++byte) {
          const value = testCase.weight[
            sourceGroup + word * VALUES_PER_WORD + byte
          ];
          const quantized = absmax === 0
            ? 0
            : Math.max(-127, Math.min(127, Math.round(value * 127 / absmax)));
          packed |= (quantized & 0xff) << (byte * 8);
        }
        words[packedRow + group * 32 + word] = packed >>> 0;
      }
    }
  }
  return { words, scales, wordsPerRow, groupsPerRow };
}

function validatePackedDispatch(testCase, packed, fixed32Subgroups) {
  if (!fixed32Subgroups) {
    throw new Error("Q8 S4 requires enabled fixed-size-32 subgroups");
  }
  if (testCase.rows !== 1) throw new Error("Q8 S4 supports only M == 1");
  if (testCase.formatVersion !== FORMAT_VERSION) {
    throw new Error("unsupported Q8 format version");
  }
  if (testCase.groupSize !== GROUP_SIZE) {
    throw new Error("Q8 S4 requires group size 128");
  }
  if (testCase.inner <= 0 || testCase.inner % GROUP_SIZE !== 0) {
    throw new Error("Q8 S4 requires a positive K divisible by 128");
  }
  if (packed.wordsPerRow !== testCase.inner / VALUES_PER_WORD) {
    throw new Error("Q8 packed row stride mismatch");
  }
  if (packed.groupsPerRow !== testCase.inner / GROUP_SIZE) {
    throw new Error("Q8 scale row stride mismatch");
  }
  const requiredWords =
    testCase.packedWeightOffset + testCase.columns * packed.wordsPerRow;
  const requiredScales =
    testCase.scaleOffset + testCase.columns * packed.groupsPerRow;
  if (packed.words.length < requiredWords) {
    throw new Error("Q8 packed weight buffer is too short");
  }
  if (packed.scales.length < requiredScales) {
    throw new Error("Q8 scale buffer is too short");
  }
}

function expectRejected(label, callback, pattern) {
  try {
    callback();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (!pattern.test(message)) {
      throw new Error(label + " rejected for the wrong reason: " + message);
    }
    return message;
  }
  throw new Error(label + " did not fail closed");
}

function signedByte(word, byte) {
  const value = (word >>> (byte * 8)) & 0xff;
  return value >= 128 ? value - 256 : value;
}

function referenceOutputs(testCase, packed) {
  const original = new Float32Array(testCase.columns);
  const dequantized = new Float32Array(testCase.columns);
  for (let row = 0; row < testCase.columns; ++row) {
    let originalSum = testCase.hasBias
      ? testCase.bias[testCase.biasOffset + row]
      : 0;
    let dequantizedSum = originalSum;
    const sourceRow = testCase.weightOffset + row * testCase.inner;
    const packedRow = testCase.packedWeightOffset + row * packed.wordsPerRow;
    const scaleRow = testCase.scaleOffset + row * packed.groupsPerRow;
    for (let group = 0; group < packed.groupsPerRow; ++group) {
      const scale = packed.scales[scaleRow + group];
      for (let wordInGroup = 0; wordInGroup < 32; ++wordInGroup) {
        const word = packed.words[packedRow + group * 32 + wordInGroup];
        const featureBase = group * GROUP_SIZE + wordInGroup * VALUES_PER_WORD;
        for (let byte = 0; byte < VALUES_PER_WORD; ++byte) {
          const feature = featureBase + byte;
          const inputValue = testCase.input[testCase.inputOffset + feature];
          originalSum += inputValue * testCase.weight[sourceRow + feature];
          dequantizedSum +=
            inputValue * (signedByte(word, byte) / 127) * scale;
        }
      }
    }
    original[row] = originalSum;
    dequantized[row] = dequantizedSum;
  }
  return { original, dequantized };
}

function errorMetrics(actual, expected) {
  if (actual.length !== expected.length) throw new Error("length mismatch");
  let maxAbsoluteError = 0;
  let squareError = 0;
  let squareReference = 0;
  for (let index = 0; index < actual.length; ++index) {
    if (!Number.isFinite(actual[index])) {
      throw new Error("non-finite output at " + index);
    }
    const error = actual[index] - expected[index];
    maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(error));
    squareError += error * error;
    squareReference += expected[index] * expected[index];
  }
  const rmse = Math.sqrt(squareError / Math.max(1, actual.length));
  const referenceRms = Math.sqrt(squareReference / Math.max(1, actual.length));
  return {
    maxAbsoluteError,
    rmse,
    referenceRms,
    normalizedRmse: rmse / Math.max(referenceRms, 1e-12),
  };
}

function assertKernelClose(label, actual, expected) {
  const metrics = errorMetrics(actual, expected);
  for (let index = 0; index < actual.length; ++index) {
    const allowed = 3e-4 + 1.2e-3 * Math.abs(expected[index]);
    if (Math.abs(actual[index] - expected[index]) > allowed) {
      throw new Error(
        label + " mismatch at " + index + ": actual=" + actual[index] +
        " expected=" + expected[index] + " allowed=" + allowed,
      );
    }
  }
  return metrics;
}

async function compilePipeline(device, route) {
  const source = await (await fetch(route)).text();
  const module = device.createShaderModule({ code: source });
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter((message) => message.type === "error");
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

function q8ParamsBuffer(device, testCase, packed) {
  const words = new Uint32Array(16);
  words[0] = testCase.columns;
  words[1] = testCase.inner;
  words[2] = testCase.inputOffset;
  words[3] = testCase.packedWeightOffset;
  words[4] = testCase.scaleOffset;
  words[5] = testCase.biasOffset;
  words[6] = testCase.outputOffset;
  words[7] = testCase.hasBias ? 1 : 0;
  words[8] = packed.wordsPerRow;
  words[9] = packed.groupsPerRow;
  words[10] = testCase.groupSize;
  words[11] = testCase.formatVersion;
  return gpuBuffer(device, testCase.name + " q8 params", words, GPUBufferUsage.UNIFORM);
}

function fp32ParamsBuffer(device, testCase) {
  const words = new Uint32Array(16);
  words[0] = 1;
  words[1] = testCase.columns;
  words[2] = testCase.inner;
  words[3] = testCase.inputOffset;
  words[4] = testCase.weightOffset;
  words[5] = testCase.biasOffset;
  words[6] = testCase.outputOffset;
  words[7] = testCase.hasBias ? 1 : 0;
  words[8] = testCase.inner;
  words[9] = 1;
  words[10] = testCase.inner;
  words[11] = 1;
  return gpuBuffer(device, testCase.name + " fp32 params", words, GPUBufferUsage.UNIFORM);
}

function bindGroup(device, pipeline, buffers) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((buffer, binding) => ({
      binding,
      resource: { buffer },
    })),
  });
}

async function readFloats(device, source, count) {
  const readback = device.createBuffer({
    label: "packed GEMV readback",
    size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, count * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return values;
}

function encodeDispatches(device, pipeline, group, dispatchX, repeats) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  for (let repeat = 0; repeat < repeats; ++repeat) {
    pass.dispatchWorkgroups(dispatchX, 1, 1);
  }
  pass.end();
  return encoder.finish();
}

async function runOnce(device, pipeline, group, dispatchX) {
  device.queue.submit([encodeDispatches(device, pipeline, group, dispatchX, 1)]);
  await device.queue.onSubmittedWorkDone();
}

async function measureOnce(device, pipeline, group, dispatchX, repeats) {
  const commands = encodeDispatches(device, pipeline, group, dispatchX, repeats);
  const start = performance.now();
  device.queue.submit([commands]);
  await device.queue.onSubmittedWorkDone();
  return (performance.now() - start) / repeats;
}

function makeTimestampTimer(device, scrubBytes) {
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolve = device.createBuffer({
    label: "packed GEMV timestamp resolve",
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "packed GEMV timestamp readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  // Touching an unrelated 32 MiB immediately before the measured pass makes
  // the single-dispatch result less dependent on a prior same-weight run.
  const scrubSource = device.createBuffer({
    label: "packed GEMV cache scrub source",
    size: scrubBytes,
    usage: GPUBufferUsage.COPY_SRC,
  });
  const scrubDestination = device.createBuffer({
    label: "packed GEMV cache scrub destination",
    size: scrubBytes,
    usage: GPUBufferUsage.COPY_DST,
  });
  return {
    querySet,
    resolve,
    readback,
    scrubSource,
    scrubDestination,
    scrubBytes,
  };
}

async function measureTimestampOnce(device, timer, pipeline, group, dispatchX) {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    timer.scrubSource,
    0,
    timer.scrubDestination,
    0,
    timer.scrubBytes,
  );
  const pass = encoder.beginComputePass({
    timestampWrites: {
      querySet: timer.querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(dispatchX, 1, 1);
  pass.end();
  encoder.resolveQuerySet(timer.querySet, 0, 2, timer.resolve, 0);
  encoder.copyBufferToBuffer(timer.resolve, 0, timer.readback, 0, 16);
  device.queue.submit([encoder.finish()]);
  await timer.readback.mapAsync(GPUMapMode.READ);
  const timestamps = new BigUint64Array(timer.readback.getMappedRange().slice(0));
  timer.readback.unmap();
  if (timestamps[1] <= timestamps[0]) {
    throw new Error("non-increasing WebGPU timestamp query");
  }
  // WebGPU timestamps are reported in nanoseconds.
  return Number(timestamps[1] - timestamps[0]) / 1e6;
}

function destroyTimestampTimer(timer) {
  timer.querySet.destroy();
  timer.resolve.destroy();
  timer.readback.destroy();
  timer.scrubSource.destroy();
  timer.scrubDestination.destroy();
}

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

async function exerciseCase(
  device,
  pipelines,
  testCase,
  fixed32Subgroups,
  timestampTimer,
) {
  const packed = packQ8(testCase);
  validatePackedDispatch(testCase, packed, fixed32Subgroups);
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const input = gpuBuffer(device, testCase.name + " input", testCase.input, storage);
  const fp32Weight = gpuBuffer(
    device,
    testCase.name + " fp32 weight",
    testCase.weight,
    storage,
  );
  const packedWeight = gpuBuffer(
    device,
    testCase.name + " packed weight",
    packed.words,
    storage,
  );
  const scales = gpuBuffer(device, testCase.name + " scales", packed.scales, storage);
  const bias = testCase.hasBias
    ? gpuBuffer(device, testCase.name + " bias", testCase.bias, storage)
    : input;
  const outputLength = testCase.outputOffset + testCase.columns + 3;
  const outputInitial = new Float32Array(outputLength);
  outputInitial.fill(-777.25);
  const fp32Output = gpuBuffer(
    device,
    testCase.name + " fp32 output",
    outputInitial,
    storage,
  );
  const q8Output = gpuBuffer(
    device,
    testCase.name + " q8 output",
    outputInitial,
    storage,
  );
  const fp32Params = fp32ParamsBuffer(device, testCase);
  const q8Params = q8ParamsBuffer(device, testCase, packed);
  const fp32Group = bindGroup(device, pipelines.fp32, [
    input,
    fp32Weight,
    bias,
    fp32Output,
    fp32Params,
  ]);
  const q8Group = bindGroup(device, pipelines.q8, [
    input,
    packedWeight,
    scales,
    bias,
    q8Output,
    q8Params,
  ]);
  const dispatchX = Math.ceil(testCase.columns / OUTPUTS_PER_SUBGROUP);

  await runOnce(device, pipelines.fp32, fp32Group, dispatchX);
  await runOnce(device, pipelines.q8, q8Group, dispatchX);
  const fp32All = await readFloats(device, fp32Output, outputLength);
  const q8All = await readFloats(device, q8Output, outputLength);
  for (const [label, values] of [["fp32", fp32All], ["q8", q8All]]) {
    for (let index = 0; index < testCase.outputOffset; ++index) {
      if (values[index] !== -777.25) throw new Error(label + " overwrote prefix");
    }
    for (let index = testCase.outputOffset + testCase.columns;
         index < outputLength; ++index) {
      if (values[index] !== -777.25) throw new Error(label + " overwrote suffix");
    }
  }

  const references = referenceOutputs(testCase, packed);
  const fp32Actual = fp32All.subarray(
    testCase.outputOffset,
    testCase.outputOffset + testCase.columns,
  );
  const q8Actual = q8All.subarray(
    testCase.outputOffset,
    testCase.outputOffset + testCase.columns,
  );
  const fp32KernelError = assertKernelClose(
    testCase.name + " fp32",
    fp32Actual,
    references.original,
  );
  const q8KernelError = assertKernelClose(
    testCase.name + " q8",
    q8Actual,
    references.dequantized,
  );
  const quantizationError = errorMetrics(q8Actual, references.original);
  if (quantizationError.normalizedRmse > 0.03) {
    throw new Error(
      testCase.name + " Q8 normalized RMSE exceeded 3%: " +
      quantizationError.normalizedRmse,
    );
  }

  const evidence = {
    shape: [1, testCase.inner, testCase.columns],
    groupSize: GROUP_SIZE,
    weightBytes: {
      fp32: testCase.columns * testCase.inner * 4,
      q8: packed.wordsPerRow * testCase.columns * 4 +
        packed.groupsPerRow * testCase.columns * 4,
    },
    compressionRatio:
      (testCase.columns * testCase.inner * 4) /
      (packed.wordsPerRow * testCase.columns * 4 +
        packed.groupsPerRow * testCase.columns * 4),
    correctness: { fp32KernelError, q8KernelError, quantizationError },
  };

  if (testCase.repeats > 0) {
    device.queue.submit([
      encodeDispatches(device, pipelines.fp32, fp32Group, dispatchX, 5),
      encodeDispatches(device, pipelines.q8, q8Group, dispatchX, 5),
    ]);
    await device.queue.onSubmittedWorkDone();
    const fp32Samples = [];
    const q8Samples = [];
    for (let sample = 0; sample < 9; ++sample) {
      if (sample % 2 === 0) {
        fp32Samples.push(await measureOnce(
          device, pipelines.fp32, fp32Group, dispatchX, testCase.repeats,
        ));
        q8Samples.push(await measureOnce(
          device, pipelines.q8, q8Group, dispatchX, testCase.repeats,
        ));
      } else {
        q8Samples.push(await measureOnce(
          device, pipelines.q8, q8Group, dispatchX, testCase.repeats,
        ));
        fp32Samples.push(await measureOnce(
          device, pipelines.fp32, fp32Group, dispatchX, testCase.repeats,
        ));
      }
    }
    const fp32Median = median(fp32Samples);
    const q8Median = median(q8Samples);
    evidence.benchmark = {
      repeatsPerSample: testCase.repeats,
      samples: 9,
      fp32: {
        medianMs: fp32Median,
        samplesMs: fp32Samples,
        effectiveWeightBandwidthGbps:
          evidence.weightBytes.fp32 / fp32Median / 1e6,
      },
      q8: {
        medianMs: q8Median,
        samplesMs: q8Samples,
        effectiveWeightBandwidthGbps:
          evidence.weightBytes.q8 / q8Median / 1e6,
      },
      speedup: fp32Median / q8Median,
    };
    if (timestampTimer) {
      const fp32TimestampSamples = [];
      const q8TimestampSamples = [];
      for (let sample = 0; sample < 11; ++sample) {
        if (sample % 2 === 0) {
          fp32TimestampSamples.push(await measureTimestampOnce(
            device, timestampTimer, pipelines.fp32, fp32Group, dispatchX,
          ));
          q8TimestampSamples.push(await measureTimestampOnce(
            device, timestampTimer, pipelines.q8, q8Group, dispatchX,
          ));
        } else {
          q8TimestampSamples.push(await measureTimestampOnce(
            device, timestampTimer, pipelines.q8, q8Group, dispatchX,
          ));
          fp32TimestampSamples.push(await measureTimestampOnce(
            device, timestampTimer, pipelines.fp32, fp32Group, dispatchX,
          ));
        }
      }
      const fp32TimestampMedian = median(fp32TimestampSamples);
      const q8TimestampMedian = median(q8TimestampSamples);
      evidence.benchmark.coldGpuTimestamp = {
        cacheScrubBytes: timestampTimer.scrubBytes,
        samples: 11,
        fp32: {
          medianMs: fp32TimestampMedian,
          samplesMs: fp32TimestampSamples,
        },
        q8: {
          medianMs: q8TimestampMedian,
          samplesMs: q8TimestampSamples,
        },
        speedup: fp32TimestampMedian / q8TimestampMedian,
      };
    }
  }

  for (const buffer of [
    input,
    fp32Weight,
    packedWeight,
    scales,
    fp32Output,
    q8Output,
    fp32Params,
    q8Params,
  ]) buffer.destroy();
  if (testCase.hasBias) bias.destroy();
  return evidence;
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
  result.adapter = {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
    subgroupMinSize: info.subgroupMinSize ?? null,
    subgroupMaxSize: info.subgroupMaxSize ?? null,
    subgroups: adapter.features.has("subgroups"),
    features: Array.from(adapter.features.values()).sort(),
  };
  result.userAgent = navigator.userAgent;

  // These validation probes are CPU-only and prove that unsupported layouts
  // never reach pipeline dispatch.
  const validProbe = makeCase({
    name: "validation-probe",
    inner: 128,
    columns: 4,
    seed: 1,
    repeats: 0,
  });
  const validPacked = packQ8(validProbe);
  result.failClosed = {
    wrongRows: expectRejected(
      "wrong rows",
      () => validatePackedDispatch({ ...validProbe, rows: 2 }, validPacked, true),
      /M == 1/,
    ),
    badInner: expectRejected(
      "bad inner",
      () => validatePackedDispatch({ ...validProbe, inner: 129 }, validPacked, true),
      /divisible by 128/,
    ),
    badGroup: expectRejected(
      "bad group",
      () => validatePackedDispatch({ ...validProbe, groupSize: 64 }, validPacked, true),
      /group size 128/,
    ),
    badVersion: expectRejected(
      "bad version",
      () => validatePackedDispatch({ ...validProbe, formatVersion: 2 }, validPacked, true),
      /format version/,
    ),
    noFixedSubgroup: expectRejected(
      "missing fixed subgroup",
      () => validatePackedDispatch(validProbe, validPacked, false),
      /fixed-size-32/,
    ),
    shortWeights: expectRejected(
      "short weights",
      () => validatePackedDispatch(
        validProbe,
        { ...validPacked, words: validPacked.words.subarray(0, 1) },
        true,
      ),
      /weight buffer is too short/,
    ),
    shortScales: expectRejected(
      "short scales",
      () => validatePackedDispatch(
        validProbe,
        { ...validPacked, scales: validPacked.scales.subarray(0, 1) },
        true,
      ),
      /scale buffer is too short/,
    ),
  };

  if (!fixed32Subgroups) {
    result.skipped = "adapter does not expose fixed-size-32 subgroups";
    result.ok = true;
  } else {
    const requiredFeatures = ["subgroups"];
    if (adapter.features.has("timestamp-query")) {
      requiredFeatures.push("timestamp-query");
    }
    const device = await adapter.requestDevice({ requiredFeatures });
    device.pushErrorScope("validation");
    device.addEventListener("uncapturederror", (event) => {
      result.errors.push(event.error?.message ?? String(event.error));
    });
    const pipelines = {
      fp32: await compilePipeline(device, "/linear_gemv_subgroup_s4.wgsl"),
      q8: await compilePipeline(device, "/linear_gemv_q8_s4.wgsl"),
    };
    const timestampTimer = device.features.has("timestamp-query")
      ? makeTimestampTimer(device, 32 * 1024 * 1024)
      : null;
    result.timestampQueries = timestampTimer !== null;
    const cases = [
      makeCase({
        name: "offset-tail-bias-correctness",
        inner: 256,
        columns: 7,
        inputOffset: 3,
        weightOffset: 5,
        packedWeightOffset: 7,
        scaleOffset: 3,
        biasOffset: 2,
        outputOffset: 4,
        hasBias: true,
        seed: 101,
        repeats: 0,
      }),
      makeCase({
        name: "qwen2.5-0.5b-q-proj",
        inner: 896,
        columns: 896,
        seed: 211,
        repeats: 80,
      }),
      makeCase({
        name: "qwen2.5-0.5b-gate-up-proj",
        inner: 896,
        columns: 4864,
        seed: 307,
        repeats: 30,
      }),
      makeCase({
        name: "qwen2.5-0.5b-down-proj",
        inner: 4864,
        columns: 896,
        seed: 401,
        repeats: 30,
      }),
    ];
    if (result.extended) {
      cases.push(makeCase({
        name: "qwen2.5-1.5b-gate-up-proj",
        inner: 1536,
        columns: 8960,
        seed: 503,
        repeats: 12,
      }));
    }
    result.cases = {};
    for (const testCase of cases) {
      result.cases[testCase.name] = await exerciseCase(
        device,
        pipelines,
        testCase,
        fixed32Subgroups,
        timestampTimer,
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const validationError = await device.popErrorScope();
    if (validationError) result.errors.push(validationError.message);
    if (result.errors.length !== 0) throw new Error(result.errors.join("\n"));
    if (timestampTimer) destroyTimestampTimer(timestampTimer);
    result.ok = true;
  }
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
    "http://127.0.0.1:" + address.port + "/?adapter=" + adapterMode +
      "&extended=" + (extended ? "1" : "0"),
  );
  await page.waitForFunction(() => globalThis.packedGemvTest?.done, {
    timeout: extended ? 240_000 : 120_000,
  });
  const browserResult = await page.evaluate(() => globalThis.packedGemvTest);
  console.log(JSON.stringify(browserResult, null, 2));
  if (!browserResult.ok) {
    throw new Error(browserResult.message ?? browserResult.errors.join("\n"));
  }
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
