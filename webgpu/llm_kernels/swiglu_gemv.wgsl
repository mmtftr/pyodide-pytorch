// Portable contiguous fp32 decode SwiGLU. One workgroup owns 64 output
// features and reuses each input tile across both gate and up projections.
const WORKGROUP_SIZE: u32 = 64u;
const INNER_TILE: u32 = 256u;

struct Params {
    columns: u32,
    inner: u32,
    input_offset: u32,
    gate_weight_offset: u32,
    up_weight_offset: u32,
    gate_bias_offset: u32,
    up_bias_offset: u32,
    output_offset: u32,
    has_gate_bias: u32,
    has_up_bias: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gate_weight: array<f32>;
@group(0) @binding(2) var<storage, read> up_weight: array<f32>;
@group(0) @binding(3) var<storage, read> gate_bias: array<f32>;
@group(0) @binding(4) var<storage, read> up_bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

var<workgroup> input_tile: array<f32, INNER_TILE>;

fn silu(value: f32) -> f32 {
    return value / (1.0 + exp(-value));
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    let column = workgroup_id.x * WORKGROUP_SIZE + lane;
    var gate = 0.0;
    var up = 0.0;

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
            let gate_base = params.gate_weight_offset +
                column * params.inner + inner_base;
            let up_base = params.up_weight_offset +
                column * params.inner + inner_base;
            for (var feature = 0u; feature < tile_features; feature++) {
                let input_value = input_tile[feature];
                gate += input_value * gate_weight[gate_base + feature];
                up += input_value * up_weight[up_base + feature];
            }
        }
        workgroupBarrier();
    }

    if (column < params.columns) {
        if (params.has_gate_bias != 0u) {
            gate += gate_bias[params.gate_bias_offset + column];
        }
        if (params.has_up_bias != 0u) {
            up += up_bias[params.up_bias_offset + column];
        }
        output[params.output_offset + column] = silu(gate) * up;
    }
}
