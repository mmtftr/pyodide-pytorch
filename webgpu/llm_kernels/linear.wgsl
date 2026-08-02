// Generic fp32 linear kernel for strided weights and arbitrary M/N/K tails.
const TILE: u32 = 8u;

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

var<workgroup> input_tile: array<f32, 64>;
var<workgroup> weight_tile: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = workgroup_id.y * TILE + local_id.y;
    let column = workgroup_id.x * TILE + local_id.x;
    let local_index = local_id.y * TILE + local_id.x;
    var accumulator = 0.0;

    let tile_count = (params.inner + TILE - 1u) / TILE;
    for (var tile = 0u; tile < tile_count; tile++) {
        let input_feature = tile * TILE + local_id.x;
        if (row < params.rows && input_feature < params.inner) {
            let index = params.input_offset +
                row * params.input_stride0 +
                input_feature * params.input_stride1;
            input_tile[local_index] = input[index];
        } else {
            input_tile[local_index] = 0.0;
        }

        let weight_feature = tile * TILE + local_id.y;
        if (column < params.columns && weight_feature < params.inner) {
            let index = params.weight_offset +
                column * params.weight_stride0 +
                weight_feature * params.weight_stride1;
            weight_tile[local_index] = weight[index];
        } else {
            weight_tile[local_index] = 0.0;
        }
        workgroupBarrier();

        for (var feature = 0u; feature < TILE; feature++) {
            accumulator += input_tile[local_id.y * TILE + feature] *
                weight_tile[feature * TILE + local_id.x];
        }
        workgroupBarrier();
    }

    if (row < params.rows && column < params.columns) {
        if (params.has_bias != 0u) {
            accumulator += bias[params.bias_offset + column];
        }
        output[params.output_offset + row * params.columns + column] = accumulator;
    }
}
