// Contiguous fp32 GEMV fast path for linear inputs flattened to one row.
// Each invocation owns one output dot product, avoiding the eight row lanes
// launched by the generic 8x8 GEMM kernel when M == 1.
const WORKGROUP_SIZE: u32 = 64u;
const INNER_TILE: u32 = 256u;

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

var<workgroup> input_tile: array<f32, INNER_TILE>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    let column = workgroup_id.x * WORKGROUP_SIZE + lane;
    var accumulator = 0.0;

    for (var inner_base = 0u; inner_base < params.inner; inner_base += INNER_TILE) {
        for (var tile_feature = lane;
             tile_feature < INNER_TILE;
             tile_feature += WORKGROUP_SIZE) {
            let feature = inner_base + tile_feature;
            if (feature < params.inner) {
                input_tile[tile_feature] = input[params.input_offset + feature];
            } else {
                input_tile[tile_feature] = 0.0;
            }
        }
        workgroupBarrier();

        if (column < params.columns) {
            let tile_features = min(INNER_TILE, params.inner - inner_base);
            let weight_base =
                params.weight_offset + column * params.inner + inner_base;
            for (var feature = 0u; feature < tile_features; feature++) {
                accumulator +=
                    input_tile[feature] * weight[weight_base + feature];
            }
        }
        workgroupBarrier();
    }

    if (column < params.columns) {
        if (params.has_bias != 0u) {
            accumulator += bias[params.bias_offset + column];
        }
        output[params.output_offset + column] = accumulator;
    }
}
