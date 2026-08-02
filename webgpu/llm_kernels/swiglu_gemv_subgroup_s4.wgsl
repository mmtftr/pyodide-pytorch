// Contiguous fp32 decode SwiGLU for devices with a fixed subgroup size of 32.
// One subgroup reduces gate and up dot products for four output features.
enable subgroups;

const SUBGROUP_SIZE: u32 = 32u;
const OUTPUTS_PER_SUBGROUP: u32 = 4u;

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

fn silu(value: f32) -> f32 {
    return value / (1.0 + exp(-value));
}

@compute @workgroup_size(SUBGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32) {
    let first_column = workgroup_id.x * OUTPUTS_PER_SUBGROUP;
    var gate = vec4<f32>(0.0);
    var up = vec4<f32>(0.0);

    for (var inner_base = 0u;
         inner_base < params.inner;
         inner_base += SUBGROUP_SIZE) {
        let feature = inner_base + lane;
        if (feature < params.inner) {
            let input_value = input[params.input_offset + feature];
            if (first_column < params.columns) {
                gate.x += input_value * gate_weight[
                    params.gate_weight_offset +
                    first_column * params.inner + feature
                ];
                up.x += input_value * up_weight[
                    params.up_weight_offset +
                    first_column * params.inner + feature
                ];
            }
            if (first_column + 1u < params.columns) {
                gate.y += input_value * gate_weight[
                    params.gate_weight_offset +
                    (first_column + 1u) * params.inner + feature
                ];
                up.y += input_value * up_weight[
                    params.up_weight_offset +
                    (first_column + 1u) * params.inner + feature
                ];
            }
            if (first_column + 2u < params.columns) {
                gate.z += input_value * gate_weight[
                    params.gate_weight_offset +
                    (first_column + 2u) * params.inner + feature
                ];
                up.z += input_value * up_weight[
                    params.up_weight_offset +
                    (first_column + 2u) * params.inner + feature
                ];
            }
            if (first_column + 3u < params.columns) {
                gate.w += input_value * gate_weight[
                    params.gate_weight_offset +
                    (first_column + 3u) * params.inner + feature
                ];
                up.w += input_value * up_weight[
                    params.up_weight_offset +
                    (first_column + 3u) * params.inner + feature
                ];
            }
        }
    }

    gate = subgroupAdd(gate);
    up = subgroupAdd(up);
    if (lane == 0u) {
        for (var output_index = 0u;
             output_index < OUTPUTS_PER_SUBGROUP;
             output_index++) {
            let column = first_column + output_index;
            if (column < params.columns) {
                var gate_value = gate[output_index];
                var up_value = up[output_index];
                if (params.has_gate_bias != 0u) {
                    gate_value += gate_bias[
                        params.gate_bias_offset + column
                    ];
                }
                if (params.has_up_bias != 0u) {
                    up_value += up_bias[params.up_bias_offset + column];
                }
                output[params.output_offset + column] =
                    silu(gate_value) * up_value;
            }
        }
    }
}
