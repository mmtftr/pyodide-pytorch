// Contiguous fp32 M == 1 GEMV for devices with a fixed subgroup size of 32.
// One subgroup cooperatively reduces four output dot products. For each input
// feature, a lane reuses its input value across four adjacent weight rows.
enable subgroups;

const SUBGROUP_SIZE: u32 = 32u;
const OUTPUTS_PER_SUBGROUP: u32 = 4u;

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

@compute @workgroup_size(SUBGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32) {
    let first_column = workgroup_id.x * OUTPUTS_PER_SUBGROUP;
    var accumulator = vec4<f32>(0.0);

    for (var inner_base = 0u;
         inner_base < params.inner;
         inner_base += SUBGROUP_SIZE) {
        let feature = inner_base + lane;
        if (feature < params.inner) {
            let input_value = input[params.input_offset + feature];
            if (first_column < params.columns) {
                accumulator.x += input_value * weight[
                    params.weight_offset + first_column * params.inner + feature
                ];
            }
            if (first_column + 1u < params.columns) {
                accumulator.y += input_value * weight[
                    params.weight_offset +
                    (first_column + 1u) * params.inner + feature
                ];
            }
            if (first_column + 2u < params.columns) {
                accumulator.z += input_value * weight[
                    params.weight_offset +
                    (first_column + 2u) * params.inner + feature
                ];
            }
            if (first_column + 3u < params.columns) {
                accumulator.w += input_value * weight[
                    params.weight_offset +
                    (first_column + 3u) * params.inner + feature
                ];
            }
        }
    }

    accumulator = subgroupAdd(accumulator);
    if (lane == 0u) {
        for (var output_index = 0u;
             output_index < OUTPUTS_PER_SUBGROUP;
             output_index++) {
            let column = first_column + output_index;
            if (column < params.columns) {
                var value = accumulator[output_index];
                if (params.has_bias != 0u) {
                    value += bias[params.bias_offset + column];
                }
                output[params.output_offset + column] = value;
            }
        }
    }
}
