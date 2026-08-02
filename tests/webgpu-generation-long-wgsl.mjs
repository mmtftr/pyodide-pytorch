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
  "any_bool.wgsl",
  "bitwise_not_bool.wgsl",
  "bool_to_long.wgsl",
  "long_cumsum.wgsl",
  "long_isin.wgsl",
  "long_lt_scalar.wgsl",
  "mul_bool_tensor.wgsl",
  "ne_tensor.wgsl",
];

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
for (const name of shaderNames) {
  const filename = path.join(shaderDirectory, name);
  if (!fs.existsSync(filename)) {
    throw new Error("shader does not exist: " + filename);
  }
}

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
globalThis.generationLongTest = result;

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
    u32: new Uint32Array(bytes),
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
  const module = device.createShaderModule({
    label: name,
    code: await response.text(),
  });
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

async function mappedCopy(device, source, byteLength) {
  const readback = device.createBuffer({
    label: "generation Long readback",
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

function writeCanonicalLong(words, elementIndex, value) {
  if (!Number.isInteger(value) || value < -2147483648 || value > 2147483647) {
    throw new Error("test value is outside restricted Long: " + value);
  }
  const word = elementIndex * 2;
  words[word] = value >>> 0;
  words[word + 1] = value < 0 ? 0xffffffff : 0;
}

function readLong(words, elementIndex) {
  const word = elementIndex * 2;
  let value = (BigInt(words[word + 1]) << 32n) | BigInt(words[word]);
  if ((words[word + 1] & 0x80000000) !== 0) value -= 1n << 64n;
  return value;
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(label + ": " + String(actual) + " != " + String(expected));
  }
}

async function runCumsum(device, pipelineValue) {
  const rows = [
    [1, 0, 1, 1, 0, 1, 1],
    [-2, 5, -3, 0, 1, -1, 2],
    [2147483647, -1, -2147483646, -1, 1, -1, 1],
  ];
  const rowCount = rows.length;
  const columns = rows[0].length;
  const inputOffset = 2;
  const outputOffset = 1;
  const elements = rowCount * columns;
  const inputWords = new Uint32Array((inputOffset + elements + 2) * 2);
  inputWords.fill(0x5a5a5a5a);
  rows.flat().forEach((value, index) => {
    writeCanonicalLong(inputWords, inputOffset + index, value);
  });
  const outputWords = new Uint32Array((outputOffset + elements + 2) * 2);
  outputWords.fill(0xa5a5a5a5);

  const input = gpuBuffer(
    device,
    "cumsum input",
    inputWords,
    GPUBufferUsage.STORAGE,
  );
  const output = gpuBuffer(
    device,
    "cumsum output",
    outputWords,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const workgroups = Math.ceil(rowCount / 64);
  const dispatchX = Math.min(workgroups, 65535);
  const params = uniform(device, "cumsum params", 32);
  params.u32.set([
    rowCount,
    columns,
    inputOffset,
    outputOffset,
    dispatchX,
  ]);

  const encoder = device.createCommandEncoder();
  const compute = encoder.beginComputePass();
  compute.setPipeline(pipelineValue);
  compute.setBindGroup(
    0,
    bindGroup(device, pipelineValue, [input, output, params.finish()]),
  );
  compute.dispatchWorkgroups(dispatchX, Math.ceil(workgroups / dispatchX));
  compute.end();
  device.queue.submit([encoder.finish()]);

  const actualWords = new Uint32Array(
    await mappedCopy(device, output, outputWords.byteLength),
  );
  let checked = 0;
  for (let row = 0; row < rowCount; ++row) {
    let sum = 0n;
    for (let column = 0; column < columns; ++column) {
      const linear = row * columns + column;
      sum += BigInt(rows[row][column]);
      assertEqual(
        readLong(actualWords, outputOffset + linear),
        sum,
        "cumsum row " + row + " column " + column,
      );
      const low = actualWords[(outputOffset + linear) * 2];
      const high = actualWords[(outputOffset + linear) * 2 + 1];
      assertEqual(
        high,
        (low & 0x80000000) !== 0 ? 0xffffffff : 0,
        "cumsum canonical high word " + linear,
      );
      checked += 1;
    }
  }
  assertEqual(actualWords[0], 0xa5a5a5a5, "cumsum prefix bound low");
  assertEqual(actualWords[1], 0xa5a5a5a5, "cumsum prefix bound high");
  const suffixWord = (outputOffset + elements) * 2;
  assertEqual(actualWords[suffixWord], 0xa5a5a5a5, "cumsum suffix bound low");
  assertEqual(actualWords[suffixWord + 1], 0xa5a5a5a5, "cumsum suffix bound high");
  return { dispatches: 1, rows: rowCount, columns, valuesChecked: checked };
}

function elementStorageIndex(linear, sizes, strides, offset) {
  let remaining = linear;
  let index = offset;
  for (let dim = sizes.length - 1; dim >= 0; --dim) {
    const coordinate = remaining % sizes[dim];
    remaining = Math.floor(remaining / sizes[dim]);
    index += coordinate * strides[dim];
  }
  return index;
}

async function runBoolToLong(device, pipelineValue) {
  const rankEightSource = new Uint8Array(140);
  const bytePattern = [0, 1, 2, 255, 0, 17, 128];
  for (let index = 0; index < rankEightSource.length; ++index) {
    rankEightSource[index] = bytePattern[index % bytePattern.length];
  }
  const testCases = [
    {
      name: "Bool to Long scalar offset",
      sizes: [],
      strides: [],
      sourceOffset: 5,
      sourceBytes: new Uint8Array([0, 0, 0, 0, 0, 128, 0, 0]),
    },
    {
      name: "Bool to Long rank-eight strided tail",
      sizes: [1, 1, 1, 1, 1, 1, 2, 67],
      strides: [139, 137, 131, 127, 113, 109, 0, 2],
      sourceOffset: 3,
      sourceBytes: rankEightSource,
    },
    {
      name: "Bool to Long empty",
      sizes: [2, 0, 3],
      strides: [9, 3, 1],
      sourceOffset: 0,
      sourceBytes: new Uint8Array(4),
    },
  ];

  let dispatches = 0;
  let valuesChecked = 0;
  const summaries = {};
  for (const testCase of testCases) {
    const length = testCase.sizes.reduce((value, size) => value * size, 1);
    const source = gpuBuffer(
      device,
      testCase.name + " input",
      testCase.sourceBytes,
      GPUBufferUsage.STORAGE,
    );
    const initialWords = new Uint32Array(length * 2 + 2);
    initialWords.fill(0xdecafbad);
    const output = gpuBuffer(
      device,
      testCase.name + " output",
      initialWords,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );

    // The production C++ path skips an empty dispatch. Dispatch one raw
    // workgroup here as an additional shader-side bounds check.
    const workgroups = Math.max(1, Math.ceil(length / 64));
    const dispatchX = Math.min(workgroups, 65535);
    const params = uniform(device, testCase.name + " params", 80);
    params.u32.set([
      length,
      testCase.sizes.length,
      testCase.sourceOffset,
      dispatchX,
    ]);
    params.u32.set(testCase.sizes, 4);
    params.u32.set(testCase.strides, 12);

    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(pipelineValue);
    compute.setBindGroup(
      0,
      bindGroup(device, pipelineValue, [source, output, params.finish()]),
    );
    compute.dispatchWorkgroups(
      dispatchX,
      Math.ceil(workgroups / dispatchX),
    );
    compute.end();
    device.queue.submit([encoder.finish()]);
    dispatches += 1;

    const actualWords = new Uint32Array(
      await mappedCopy(device, output, initialWords.byteLength),
    );
    for (let linear = 0; linear < length; ++linear) {
      const sourceIndex = elementStorageIndex(
        linear,
        testCase.sizes,
        testCase.strides,
        testCase.sourceOffset,
      );
      const expected = testCase.sourceBytes[sourceIndex] === 0 ? 0n : 1n;
      assertEqual(
        readLong(actualWords, linear),
        expected,
        testCase.name + " value " + linear,
      );
      assertEqual(
        actualWords[linear * 2 + 1],
        0,
        testCase.name + " canonical high word " + linear,
      );
      valuesChecked += 1;
    }
    assertEqual(
      actualWords[length * 2],
      0xdecafbad,
      testCase.name + " suffix bound low",
    );
    assertEqual(
      actualWords[length * 2 + 1],
      0xdecafbad,
      testCase.name + " suffix bound high",
    );
    summaries[testCase.name] = { length, rank: testCase.sizes.length };
  }
  return { dispatches, valuesChecked, cases: summaries };
}

async function runIsin(device, pipelineValue, invert) {
  const sizes = [3, 5];
  const strides = [7, 1];
  const elementsOffset = 2;
  const values = [-2, 3, 7, 42, 0, 9, -2, 99, 4, 7, 1, 2, 3, 4, 5];
  const maximumElementIndex = elementStorageIndex(
    values.length - 1,
    sizes,
    strides,
    elementsOffset,
  );
  const elementWords = new Uint32Array((maximumElementIndex + 3) * 2);
  elementWords.fill(0x6b6b6b6b);
  values.forEach((value, linear) => {
    writeCanonicalLong(
      elementWords,
      elementStorageIndex(linear, sizes, strides, elementsOffset),
      value,
    );
  });
  // A noncanonical Long must not match the otherwise-identical test value.
  const noncanonicalLinear = 7;
  const noncanonicalStorage = elementStorageIndex(
    noncanonicalLinear,
    sizes,
    strides,
    elementsOffset,
  );
  elementWords[noncanonicalStorage * 2 + 1] = 0x12345678;

  const testValues = [-2, 7, 42, 99];
  const testOffset = 1;
  const testStride = 2;
  const testWords = new Uint32Array(20);
  testWords.fill(0x7c7c7c7c);
  testValues.forEach((value, index) => {
    writeCanonicalLong(testWords, testOffset + index * testStride, value);
  });
  // The 99 test entry is also noncanonical and therefore cannot match.
  testWords[(testOffset + 3 * testStride) * 2 + 1] = 0x87654321;

  const elements = gpuBuffer(
    device,
    "isin strided elements",
    elementWords,
    GPUBufferUsage.STORAGE,
  );
  const testElements = gpuBuffer(
    device,
    "isin strided test elements",
    testWords,
    GPUBufferUsage.STORAGE,
  );
  const outputByteLength = Math.ceil(values.length / 4) * 4 + 4;
  const outputInitial = new Uint8Array(outputByteLength).fill(0xcd);
  const output = gpuBuffer(
    device,
    "isin packed Bool output",
    outputInitial,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const outputWords = Math.ceil(values.length / 4);
  const workgroups = Math.ceil(outputWords / 64);
  const dispatchX = Math.min(workgroups, 65535);
  const params = uniform(device, "isin params", 96);
  params.u32.set([
    values.length,
    sizes.length,
    elementsOffset,
    testValues.length,
    testOffset,
    testStride,
    invert ? 1 : 0,
    dispatchX,
  ]);
  params.u32.set(sizes, 8);
  params.u32.set(strides, 16);

  const encoder = device.createCommandEncoder();
  const compute = encoder.beginComputePass();
  compute.setPipeline(pipelineValue);
  compute.setBindGroup(
    0,
    bindGroup(device, pipelineValue, [
      elements,
      testElements,
      output,
      params.finish(),
    ]),
  );
  compute.dispatchWorkgroups(dispatchX, Math.ceil(workgroups / dispatchX));
  compute.end();
  device.queue.submit([encoder.finish()]);

  const actual = new Uint8Array(
    await mappedCopy(device, output, outputByteLength),
  );
  const canonicalTest = new Set([-2, 7, 42]);
  values.forEach((value, index) => {
    const member = index !== noncanonicalLinear && canonicalTest.has(value);
    assertEqual(
      actual[index],
      (invert ? !member : member) ? 1 : 0,
      "isin byte " + index + " invert=" + invert,
    );
  });
  for (let index = values.length; index < outputWords * 4; ++index) {
    assertEqual(actual[index], 0, "isin canonical tail byte " + index);
  }
  for (let index = outputWords * 4; index < outputByteLength; ++index) {
    assertEqual(actual[index], 0xcd, "isin output bound byte " + index);
  }
  return {
    dispatches: 1,
    invert,
    valuesChecked: values.length,
    testElements: testValues.length,
  };
}

async function runAny(device, pipelineValue) {
  const sizes = [17, 19];
  const strides = [23, 1];
  const inputOffset = 3;
  const length = sizes[0] * sizes[1];
  const maximumIndex = elementStorageIndex(
    length - 1,
    sizes,
    strides,
    inputOffset,
  );
  const storageLength = (maximumIndex + 4) & ~3;
  const emptyInput = new Uint8Array(storageLength);

  async function runCase(expected, trueLinearIndex) {
    const inputBytes = emptyInput.slice();
    if (trueLinearIndex !== null) {
      inputBytes[elementStorageIndex(
        trueLinearIndex,
        sizes,
        strides,
        inputOffset,
      )] = 1;
    }
    const input = gpuBuffer(
      device,
      "any strided Bool input",
      inputBytes,
      GPUBufferUsage.STORAGE,
    );
    const output = gpuBuffer(
      device,
      "any scalar Bool output",
      new Uint8Array(8).fill(0xcc),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const params = uniform(device, "any Bool params", 80);
    params.u32.set([length, sizes.length, inputOffset], 0);
    params.u32.set(sizes, 4);
    params.u32.set(strides, 12);

    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(pipelineValue);
    compute.setBindGroup(
      0,
      bindGroup(device, pipelineValue, [input, output, params.finish()]),
    );
    compute.dispatchWorkgroups(1);
    compute.end();
    device.queue.submit([encoder.finish()]);
    const actual = new Uint8Array(await mappedCopy(device, output, 8));
    assertEqual(actual[0], expected, "any scalar value");
    for (let index = 1; index < 4; ++index) {
      assertEqual(actual[index], 0, "any canonical scalar padding " + index);
    }
    for (let index = 4; index < 8; ++index) {
      assertEqual(actual[index], 0xcc, "any output bound " + index);
    }
  }

  await runCase(0, null);
  // The last logical element is beyond the first 256 lane-strided reads.
  await runCase(1, length - 1);
  return { dispatches: 2, valuesChecked: length * 2 };
}

async function runLongLtScalar(device, pipelineValue) {
  const sizes = [3, 5];
  const strides = [7, 1];
  const inputOffset = 2;
  const values = [
    -2147483648, -2, -1, 0, 1,
    2, 7, -7, 2147483647, -1,
    4, -3, 3, 9, -9,
  ];
  const inputWords = new Uint32Array(48).fill(0x5a5a5a5a);
  values.forEach((value, linearIndex) => {
    writeCanonicalLong(
      inputWords,
      elementStorageIndex(
        linearIndex,
        sizes,
        strides,
        inputOffset,
      ),
      value,
    );
  });
  const noncanonicalLinearIndex = 7;
  const noncanonicalStorageIndex = elementStorageIndex(
    noncanonicalLinearIndex,
    sizes,
    strides,
    inputOffset,
  );
  inputWords[noncanonicalStorageIndex * 2 + 1] = 0x12345678;

  const input = gpuBuffer(
    device,
    "lt.Scalar strided Long input",
    inputWords,
    GPUBufferUsage.STORAGE,
  );
  const logicalOutputBytes = Math.ceil(values.length / 4) * 4;
  const outputByteLength = logicalOutputBytes + 4;
  const output = gpuBuffer(
    device,
    "lt.Scalar packed Bool output",
    new Uint8Array(outputByteLength).fill(0xdd),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const params = uniform(device, "Long lt.Scalar params", 96);
  params.u32.set([
    values.length,
    sizes.length,
    inputOffset,
    0, // signed-int32 scalar zero
    1, // dispatch x
  ]);
  params.u32.set(sizes, 8);
  params.u32.set(strides, 16);

  const encoder = device.createCommandEncoder();
  const compute = encoder.beginComputePass();
  compute.setPipeline(pipelineValue);
  compute.setBindGroup(
    0,
    bindGroup(device, pipelineValue, [input, output, params.finish()]),
  );
  compute.dispatchWorkgroups(1);
  compute.end();
  device.queue.submit([encoder.finish()]);
  const actual = new Uint8Array(
    await mappedCopy(device, output, outputByteLength),
  );
  values.forEach((value, index) => {
    assertEqual(
      actual[index],
      index !== noncanonicalLinearIndex && value < 0 ? 1 : 0,
      "Long lt.Scalar byte " + index,
    );
  });
  for (let index = values.length; index < logicalOutputBytes; ++index) {
    assertEqual(actual[index], 0, "Long lt.Scalar canonical tail " + index);
  }
  for (let index = logicalOutputBytes; index < outputByteLength; ++index) {
    assertEqual(actual[index], 0xdd, "Long lt.Scalar output bound " + index);
  }
  return {
    dispatches: 1,
    valuesChecked: values.length,
    noncanonicalContained: true,
  };
}

async function runBitwiseNotBool(device, pipelineValue) {
  async function runCase({
    name,
    sizes,
    strides,
    inputOffset,
    values,
  }) {
    const length = values.length;
    const maximumIndex = length === 0
      ? inputOffset
      : elementStorageIndex(
          length - 1,
          sizes,
          strides,
          inputOffset,
        );
    const inputByteLength = Math.max(4, (maximumIndex + 4) & ~3);
    const inputBytes = new Uint8Array(inputByteLength).fill(0x7e);
    values.forEach((value, linearIndex) => {
      inputBytes[elementStorageIndex(
        linearIndex,
        sizes,
        strides,
        inputOffset,
      )] = value;
    });
    const logicalOutputBytes = Math.ceil(length / 4) * 4;
    const outputByteLength = Math.max(4, logicalOutputBytes) + 4;
    const input = gpuBuffer(
      device,
      name + " input",
      inputBytes,
      GPUBufferUsage.STORAGE,
    );
    const output = gpuBuffer(
      device,
      name + " output",
      new Uint8Array(outputByteLength).fill(0xee),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const params = uniform(device, name + " params", 80);
    params.u32.set([length, sizes.length, inputOffset, 1]);
    params.u32.set(sizes, 4);
    params.u32.set(strides, 12);

    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(pipelineValue);
    compute.setBindGroup(
      0,
      bindGroup(device, pipelineValue, [input, output, params.finish()]),
    );
    compute.dispatchWorkgroups(1);
    compute.end();
    device.queue.submit([encoder.finish()]);
    const actual = new Uint8Array(
      await mappedCopy(device, output, outputByteLength),
    );
    values.forEach((value, index) => {
      assertEqual(
        actual[index],
        value === 0 ? 1 : 0,
        name + " byte " + index,
      );
    });
    for (let index = length; index < logicalOutputBytes; ++index) {
      assertEqual(actual[index], 0, name + " canonical tail " + index);
    }
    const guardStart = Math.max(4, logicalOutputBytes);
    for (let index = guardStart; index < outputByteLength; ++index) {
      assertEqual(actual[index], 0xee, name + " output bound " + index);
    }
    return length;
  }

  const scalarChecked = await runCase({
    name: "bitwise_not scalar Bool",
    sizes: [],
    strides: [],
    inputOffset: 3,
    values: [1],
  });
  const stridedChecked = await runCase({
    name: "bitwise_not strided Bool",
    sizes: [3, 5],
    strides: [7, 1],
    inputOffset: 2,
    // Nonzero bytes are logically true and must normalize to canonical false.
    values: [0, 1, 255, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
  });
  return {
    dispatches: 2,
    scalarValuesChecked: scalarChecked,
    stridedValuesChecked: stridedChecked,
  };
}

async function runMulBoolTensor(device, pipelineValue) {
  async function runScalarCase(name, lhsValue, rhsValue) {
    const lhsOffset = 3;
    const rhsOffset = 2;
    const lhsBytes = new Uint8Array(8).fill(0x7a);
    const rhsBytes = new Uint8Array(8).fill(0x6b);
    lhsBytes[lhsOffset] = lhsValue;
    rhsBytes[rhsOffset] = rhsValue;
    const lhs = gpuBuffer(
      device,
      name + " lhs",
      lhsBytes,
      GPUBufferUsage.STORAGE,
    );
    const rhs = gpuBuffer(
      device,
      name + " rhs",
      rhsBytes,
      GPUBufferUsage.STORAGE,
    );
    const outputByteLength = 8;
    const output = gpuBuffer(
      device,
      name + " output",
      new Uint8Array(outputByteLength).fill(0xef),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const params = uniform(device, name + " params", 192);
    params.u32.set([
      1, // output length
      0, // output rank
      0, // lhs rank
      0, // rhs rank
      lhsOffset,
      rhsOffset,
      1, // dispatch x
      0,
    ]);

    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(pipelineValue);
    compute.setBindGroup(
      0,
      bindGroup(device, pipelineValue, [
        lhs,
        rhs,
        output,
        params.finish(),
      ]),
    );
    compute.dispatchWorkgroups(1);
    compute.end();
    device.queue.submit([encoder.finish()]);

    const actual = new Uint8Array(
      await mappedCopy(device, output, outputByteLength),
    );
    assertEqual(
      actual[0],
      lhsValue !== 0 && rhsValue !== 0 ? 1 : 0,
      name + " result",
    );
    for (let index = 1; index < 4; ++index) {
      assertEqual(actual[index], 0, name + " canonical scalar padding " + index);
    }
    for (let index = 4; index < outputByteLength; ++index) {
      assertEqual(actual[index], 0xef, name + " output bound " + index);
    }
  }

  await runScalarCase("mul.Tensor scalar Bool false false", 0, 0);
  await runScalarCase("mul.Tensor scalar Bool false true", 0, 255);
  await runScalarCase("mul.Tensor scalar Bool true false", 255, 0);
  await runScalarCase("mul.Tensor scalar Bool true true", 255, 7);

  const name = "mul.Tensor broadcast strided Bool";
  const outputSizes = [2, 5, 3];
  const lhsSizes = [2, 1, 3];
  const lhsStrides = [7, 3, 1];
  const lhsOffset = 2;
  const lhsValues = [0, 1, 255, 7, 0, 1];
  const rhsSizes = [1, 5, 1];
  const rhsStrides = [8, 1, 1];
  const rhsOffset = 1;
  const rhsValues = [255, 0, 3, 1, 0];
  const outputLength = outputSizes.reduce((product, size) => product * size, 1);
  const lhsMaximumIndex = elementStorageIndex(
    lhsValues.length - 1,
    lhsSizes,
    lhsStrides,
    lhsOffset,
  );
  const rhsMaximumIndex = elementStorageIndex(
    rhsValues.length - 1,
    rhsSizes,
    rhsStrides,
    rhsOffset,
  );
  const lhsBytes = new Uint8Array((lhsMaximumIndex + 4) & ~3).fill(0x7c);
  const rhsBytes = new Uint8Array((rhsMaximumIndex + 4) & ~3).fill(0x6d);
  lhsValues.forEach((value, linearIndex) => {
    lhsBytes[elementStorageIndex(
      linearIndex,
      lhsSizes,
      lhsStrides,
      lhsOffset,
    )] = value;
  });
  rhsValues.forEach((value, linearIndex) => {
    rhsBytes[elementStorageIndex(
      linearIndex,
      rhsSizes,
      rhsStrides,
      rhsOffset,
    )] = value;
  });
  const lhs = gpuBuffer(device, name + " lhs", lhsBytes, GPUBufferUsage.STORAGE);
  const rhs = gpuBuffer(device, name + " rhs", rhsBytes, GPUBufferUsage.STORAGE);
  const logicalOutputBytes = Math.ceil(outputLength / 4) * 4;
  const outputByteLength = logicalOutputBytes + 4;
  const output = gpuBuffer(
    device,
    name + " output",
    new Uint8Array(outputByteLength).fill(0xed),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  );
  const params = uniform(device, name + " params", 192);
  params.u32.set([
    outputLength,
    outputSizes.length,
    lhsSizes.length,
    rhsSizes.length,
    lhsOffset,
    rhsOffset,
    1, // dispatch x
    0,
  ]);
  params.u32.set(outputSizes, 8);
  params.u32.set(lhsSizes, 16);
  params.u32.set(lhsStrides, 24);
  params.u32.set(rhsSizes, 32);
  params.u32.set(rhsStrides, 40);

  const encoder = device.createCommandEncoder();
  const compute = encoder.beginComputePass();
  compute.setPipeline(pipelineValue);
  compute.setBindGroup(
    0,
    bindGroup(device, pipelineValue, [lhs, rhs, output, params.finish()]),
  );
  compute.dispatchWorkgroups(1);
  compute.end();
  device.queue.submit([encoder.finish()]);

  const actual = new Uint8Array(
    await mappedCopy(device, output, outputByteLength),
  );
  let linearIndex = 0;
  for (let lhsRow = 0; lhsRow < 2; ++lhsRow) {
    for (let rhsColumn = 0; rhsColumn < 5; ++rhsColumn) {
      for (let lhsColumn = 0; lhsColumn < 3; ++lhsColumn) {
        const expected =
          lhsValues[lhsRow * 3 + lhsColumn] !== 0 &&
          rhsValues[rhsColumn] !== 0
            ? 1
            : 0;
        assertEqual(actual[linearIndex], expected, name + " byte " + linearIndex);
        linearIndex += 1;
      }
    }
  }
  for (let index = outputLength; index < logicalOutputBytes; ++index) {
    assertEqual(actual[index], 0, "mul.Tensor canonical tail " + index);
  }
  for (let index = logicalOutputBytes; index < outputByteLength; ++index) {
    assertEqual(actual[index], 0xed, "mul.Tensor output bound " + index);
  }
  return {
    dispatches: 5,
    scalarValuesChecked: 4,
    broadcastValuesChecked: outputLength,
    noncanonicalTrueBytes: true,
  };
}

function logicalElementCount(sizes) {
  return sizes.reduce((product, size) => product * size, 1);
}

function broadcastLogicalIndex(linearIndex, outputSizes, inputSizes) {
  const coordinates = new Array(outputSizes.length);
  let remaining = linearIndex;
  for (let dim = outputSizes.length - 1; dim >= 0; --dim) {
    coordinates[dim] = remaining % outputSizes[dim];
    remaining = Math.floor(remaining / outputSizes[dim]);
  }
  const rankOffset = outputSizes.length - inputSizes.length;
  let inputIndex = 0;
  for (let dim = 0; dim < inputSizes.length; ++dim) {
    const coordinate = inputSizes[dim] === 1
      ? 0
      : coordinates[rankOffset + dim];
    inputIndex = inputIndex * inputSizes[dim] + coordinate;
  }
  return inputIndex;
}

async function runNeTensor(device, pipelineValue) {
  function comparisonInput(name, kind, sizes, strides, offset, values) {
    const maximumIndex = elementStorageIndex(
      values.length - 1,
      sizes,
      strides,
      offset,
    );
    if (kind === "long") {
      const words = new Uint32Array((maximumIndex + 1) * 2);
      words.fill(0x5a5a5a5a);
      values.forEach((value, linearIndex) => {
        writeCanonicalLong(
          words,
          elementStorageIndex(linearIndex, sizes, strides, offset),
          value,
        );
      });
      return gpuBuffer(device, name, words, GPUBufferUsage.STORAGE);
    }
    const words = new Uint32Array(maximumIndex + 1).fill(0x6b6b6b6b);
    values.forEach((value, linearIndex) => {
      words[elementStorageIndex(linearIndex, sizes, strides, offset)] =
        value >>> 0;
    });
    return gpuBuffer(device, name, words, GPUBufferUsage.STORAGE);
  }

  async function runCase({
    name,
    outputSizes,
    lhsKind,
    lhsSizes,
    lhsStrides,
    lhsOffset,
    lhsValues,
    rhsKind,
    rhsSizes,
    rhsStrides,
    rhsOffset,
    rhsValues,
  }) {
    const outputLength = logicalElementCount(outputSizes);
    const lhs = comparisonInput(
      name + " lhs",
      lhsKind,
      lhsSizes,
      lhsStrides,
      lhsOffset,
      lhsValues,
    );
    const rhs = comparisonInput(
      name + " rhs",
      rhsKind,
      rhsSizes,
      rhsStrides,
      rhsOffset,
      rhsValues,
    );
    const logicalOutputBytes = Math.ceil(outputLength / 4) * 4;
    const outputByteLength = logicalOutputBytes + 4;
    const output = gpuBuffer(
      device,
      name + " output",
      new Uint8Array(outputByteLength).fill(0xea),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    );
    const outputWords = Math.ceil(outputLength / 4);
    const workgroups = Math.ceil(outputWords / 64);
    const dispatchX = Math.min(workgroups, 65535);
    const inputKinds =
      (lhsKind === "long" ? 1 : 0) |
      (rhsKind === "long" ? 2 : 0);
    const params = uniform(device, name + " params", 192);
    params.u32.set([
      outputLength,
      outputSizes.length,
      lhsSizes.length,
      rhsSizes.length,
      lhsOffset,
      rhsOffset,
      inputKinds,
      dispatchX,
    ]);
    params.u32.set(outputSizes, 8);
    params.u32.set(lhsSizes, 16);
    params.u32.set(lhsStrides, 24);
    params.u32.set(rhsSizes, 32);
    params.u32.set(rhsStrides, 40);

    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(pipelineValue);
    compute.setBindGroup(
      0,
      bindGroup(device, pipelineValue, [lhs, rhs, output, params.finish()]),
    );
    compute.dispatchWorkgroups(dispatchX, Math.ceil(workgroups / dispatchX));
    compute.end();
    device.queue.submit([encoder.finish()]);

    const actual = new Uint8Array(
      await mappedCopy(device, output, outputByteLength),
    );
    for (let index = 0; index < outputLength; ++index) {
      const lhsValue = lhsValues[broadcastLogicalIndex(
        index,
        outputSizes,
        lhsSizes,
      )];
      const rhsValue = rhsValues[broadcastLogicalIndex(
        index,
        outputSizes,
        rhsSizes,
      )];
      assertEqual(actual[index], lhsValue !== rhsValue ? 1 : 0, name + " byte " + index);
    }
    for (let index = outputLength; index < logicalOutputBytes; ++index) {
      assertEqual(actual[index], 0, name + " canonical tail " + index);
    }
    for (let index = logicalOutputBytes; index < outputByteLength; ++index) {
      assertEqual(actual[index], 0xea, name + " output bound " + index);
    }
    return outputLength;
  }

  const longScalarValues = await runCase({
    name: "ne.Tensor Long scalar broadcast",
    outputSizes: [2, 5],
    lhsKind: "long",
    lhsSizes: [2, 5],
    lhsStrides: [7, 1],
    lhsOffset: 2,
    lhsValues: [0, -3, 4, 0, 7, -1, 0, 2, -9, 0],
    rhsKind: "long",
    rhsSizes: [],
    rhsStrides: [],
    rhsOffset: 1,
    rhsValues: [0],
  });
  const mixedScalarValues = await runCase({
    name: "ne.Tensor Int Long scalar",
    outputSizes: [5],
    lhsKind: "int",
    lhsSizes: [5],
    lhsStrides: [2],
    lhsOffset: 1,
    lhsValues: [-2, 0, 7, -2147483648, 2147483647],
    rhsKind: "long",
    rhsSizes: [],
    rhsStrides: [],
    rhsOffset: 2,
    rhsValues: [-2],
  });
  const intBroadcastValues = await runCase({
    name: "ne.Tensor same-dtype Int broadcast",
    outputSizes: [2, 4, 3],
    lhsKind: "int",
    lhsSizes: [2, 1, 3],
    lhsStrides: [7, 3, 1],
    lhsOffset: 2,
    lhsValues: [0, -1, 2, 3, -4, 5],
    rhsKind: "int",
    rhsSizes: [1, 4, 1],
    rhsStrides: [8, 1, 1],
    rhsOffset: 1,
    rhsValues: [0, -1, 7, 5],
  });
  return {
    dispatches: 3,
    longScalarValues,
    mixedScalarValues,
    intBroadcastValues,
  };
}

try {
  const adapter = await navigator.gpu?.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("WebGPU adapter unavailable");
  const info = adapter.info ?? {};
  result.adapter = {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
  };
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

  result.cases.cumsum = await runCumsum(
    device,
    pipelines["long_cumsum.wgsl"],
  );
  result.cases.isin = await runIsin(
    device,
    pipelines["long_isin.wgsl"],
    false,
  );
  result.cases.isinInvert = await runIsin(
    device,
    pipelines["long_isin.wgsl"],
    true,
  );
  result.cases.any = await runAny(
    device,
    pipelines["any_bool.wgsl"],
  );
  result.cases.longLtScalar = await runLongLtScalar(
    device,
    pipelines["long_lt_scalar.wgsl"],
  );
  result.cases.bitwiseNotBool = await runBitwiseNotBool(
    device,
    pipelines["bitwise_not_bool.wgsl"],
  );
  result.cases.mulBoolTensor = await runMulBoolTensor(
    device,
    pipelines["mul_bool_tensor.wgsl"],
  );
  result.cases.neTensor = await runNeTensor(
    device,
    pipelines["ne_tensor.wgsl"],
  );
  result.cases.boolToLong = await runBoolToLong(
    device,
    pipelines["bool_to_long.wgsl"],
  );

  const validationError = await device.popErrorScope();
  if (validationError) result.errors.push(validationError.message);
  if (result.errors.length !== 0) throw new Error(result.errors.join("\n"));
  result.compiled = ${JSON.stringify(shaderNames)};
  result.dispatches = 16 + result.cases.boolToLong.dispatches;
  result.ok = true;
} catch (error) {
  result.message = error instanceof Error ? error.stack : String(error);
} finally {
  result.done = true;
}
</script>`;

const server = http.createServer((request, response) => {
  const pathname = new URL(request.url, "http://127.0.0.1").pathname;
  const name = pathname.slice(1);
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
  const browserArguments = [
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-dawn-features=allow_unsafe_apis",
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
  page.on("pageerror", (error) => console.error(error.stack ?? String(error)));
  await page.goto(
    "http://127.0.0.1:" + address.port + "/?adapter=" + adapterMode,
  );
  await page.waitForFunction(() => globalThis.generationLongTest?.done, {
    timeout: 120_000,
  });
  const browserResult = await page.evaluate(() => globalThis.generationLongTest);
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
