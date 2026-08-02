// Contiguous fp32 GEMM fast path.  A 64-thread workgroup owns a 16x32 output
// tile; every invocation keeps a 2x4 register block of output values.
const THREADS_X: u32 = 8u;
const THREADS_Y: u32 = 8u;
const BLOCK_M: u32 = 16u;
const BLOCK_N: u32 = 32u;
const BLOCK_K: u32 = 16u;

struct Params {
    rows: u32,
    columns: u32,
    inner: u32,
    input_offset: u32,
    weight_offset: u32,
    bias_offset: u32,
    output_offset: u32,
    has_bias: u32,
    input_stride0: u32,
    input_stride1: u32,
    weight_stride0: u32,
    weight_stride1: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> input_tile: array<f32, BLOCK_M * BLOCK_K>;
var<workgroup> weight_tile: array<f32, BLOCK_K * BLOCK_N>;

@compute @workgroup_size(THREADS_X, THREADS_Y, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_index = local_id.y * THREADS_X + local_id.x;
    let row_base = workgroup_id.y * BLOCK_M + local_id.y * 2u;
    let column_base = workgroup_id.x * BLOCK_N + local_id.x * 4u;
    var accumulator00 = 0.0;
    var accumulator01 = 0.0;
    var accumulator02 = 0.0;
    var accumulator03 = 0.0;
    var accumulator10 = 0.0;
    var accumulator11 = 0.0;
    var accumulator12 = 0.0;
    var accumulator13 = 0.0;

    // C++ selects this shader only for exact 16x32x16 contiguous tiles.
    for (var inner_base = 0u; inner_base < params.inner; inner_base += BLOCK_K) {
        for (var load_index = local_index;
             load_index < BLOCK_M * BLOCK_K;
             load_index += THREADS_X * THREADS_Y) {
            let tile_row = load_index / BLOCK_K;
            let tile_feature = load_index % BLOCK_K;
            input_tile[load_index] = input[
                params.input_offset +
                (workgroup_id.y * BLOCK_M + tile_row) * params.inner +
                inner_base + tile_feature
            ];
        }
        for (var load_index = local_index;
             load_index < BLOCK_K * BLOCK_N;
             load_index += THREADS_X * THREADS_Y) {
            // Weight is stored [N, K]. Have adjacent invocations read along K,
            // then transpose into the [K, N] workgroup tile used below.
            let tile_column = load_index / BLOCK_K;
            let tile_feature = load_index % BLOCK_K;
            weight_tile[tile_feature * BLOCK_N + tile_column] = weight[
                params.weight_offset +
                (workgroup_id.x * BLOCK_N + tile_column) * params.inner +
                inner_base + tile_feature
            ];
        }
        workgroupBarrier();

        for (var feature = 0u; feature < BLOCK_K; feature++) {
            let input0 = input_tile[(local_id.y * 2u) * BLOCK_K + feature];
            let input1 = input_tile[(local_id.y * 2u + 1u) * BLOCK_K + feature];
            let weight_base = feature * BLOCK_N + local_id.x * 4u;
            let weight0 = weight_tile[weight_base];
            let weight1 = weight_tile[weight_base + 1u];
            let weight2 = weight_tile[weight_base + 2u];
            let weight3 = weight_tile[weight_base + 3u];
            accumulator00 += input0 * weight0;
            accumulator01 += input0 * weight1;
            accumulator02 += input0 * weight2;
            accumulator03 += input0 * weight3;
            accumulator10 += input1 * weight0;
            accumulator11 += input1 * weight1;
            accumulator12 += input1 * weight2;
            accumulator13 += input1 * weight3;
        }
        workgroupBarrier();
    }

    if (params.has_bias != 0u) {
        accumulator00 += bias[params.bias_offset + column_base];
        accumulator01 += bias[params.bias_offset + column_base + 1u];
        accumulator02 += bias[params.bias_offset + column_base + 2u];
        accumulator03 += bias[params.bias_offset + column_base + 3u];
        accumulator10 += bias[params.bias_offset + column_base];
        accumulator11 += bias[params.bias_offset + column_base + 1u];
        accumulator12 += bias[params.bias_offset + column_base + 2u];
        accumulator13 += bias[params.bias_offset + column_base + 3u];
    }
    let output_row0 = params.output_offset + row_base * params.columns + column_base;
    let output_row1 = output_row0 + params.columns;
    output[output_row0] = accumulator00;
    output[output_row0 + 1u] = accumulator01;
    output[output_row0 + 2u] = accumulator02;
    output[output_row0 + 3u] = accumulator03;
    output[output_row1] = accumulator10;
    output[output_row1 + 1u] = accumulator11;
    output[output_row1 + 2u] = accumulator12;
    output[output_row1 + 3u] = accumulator13;
}
