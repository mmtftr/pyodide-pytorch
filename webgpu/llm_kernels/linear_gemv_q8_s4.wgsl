// Experimental weight-only Q8 GEMV for fixed-size-32 subgroup devices.
//
// Format version 1 stores each contiguous weight row as little-endian groups
// of four signed int8 values in u32 words. Every 128-value row group has one
// fp32 absmax scale; unpack4x8snorm(word) * scale reconstructs the weights.
// Quantizers must clamp to [-127, 127], so both signs use the same divisor.
//
// This kernel intentionally supports only M == 1, K % 128 == 0, contiguous
// row-major packed weights, and a fixed subgroup size of 32. The host must
// validate buffer extents before dispatch. The uniform checks below make an
// invalid format/layout a no-write operation, but are not a substitute for
// validating WebGPU binding sizes on the host.
enable subgroups;

const SUBGROUP_SIZE: u32 = 32u;
const OUTPUTS_PER_SUBGROUP: u32 = 4u;
const VALUES_PER_WORD: u32 = 4u;
const GROUP_SIZE: u32 = 128u;
const WORDS_PER_GROUP: u32 = GROUP_SIZE / VALUES_PER_WORD;
const FORMAT_VERSION: u32 = 1u;

struct Params {
    columns: u32,
    inner: u32,
    input_offset: u32,
    packed_weight_offset: u32,
    scale_offset: u32,
    bias_offset: u32,
    output_offset: u32,
    has_bias: u32,
    words_per_row: u32,
    groups_per_row: u32,
    group_size: u32,
    format_version: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weight: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn valid_layout() -> bool {
    return params.format_version == FORMAT_VERSION &&
        params.group_size == GROUP_SIZE &&
        params.inner > 0u &&
        params.inner % GROUP_SIZE == 0u &&
        params.words_per_row == params.inner / VALUES_PER_WORD &&
        params.groups_per_row == params.inner / GROUP_SIZE;
}

@compute @workgroup_size(SUBGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(subgroup_invocation_id) lane: u32) {
    if (!valid_layout()) {
        return;
    }

    let first_column = workgroup_id.x * OUTPUTS_PER_SUBGROUP;
    var accumulator = vec4<f32>(0.0);

    for (var group = 0u; group < params.groups_per_row; group++) {
        // One lane loads four per-row scales, then broadcasts them. This avoids
        // reading the same 16 bytes from storage in all 32 lanes.
        var row_scales = vec4<f32>(0.0);
        if (lane == 0u) {
            if (first_column < params.columns) {
                row_scales.x = scales[
                    params.scale_offset +
                    first_column * params.groups_per_row + group
                ];
            }
            if (first_column + 1u < params.columns) {
                row_scales.y = scales[
                    params.scale_offset +
                    (first_column + 1u) * params.groups_per_row + group
                ];
            }
            if (first_column + 2u < params.columns) {
                row_scales.z = scales[
                    params.scale_offset +
                    (first_column + 2u) * params.groups_per_row + group
                ];
            }
            if (first_column + 3u < params.columns) {
                row_scales.w = scales[
                    params.scale_offset +
                    (first_column + 3u) * params.groups_per_row + group
                ];
            }
        }
        row_scales = subgroupBroadcast(row_scales, 0u);

        // K is a multiple of 128, so each lane owns one packed word and four
        // consecutive activations in every group.
        let word_in_row = group * WORDS_PER_GROUP + lane;
        let feature = group * GROUP_SIZE + lane * VALUES_PER_WORD;
        let activation = vec4<f32>(
            input[params.input_offset + feature],
            input[params.input_offset + feature + 1u],
            input[params.input_offset + feature + 2u],
            input[params.input_offset + feature + 3u]
        );

        if (first_column < params.columns) {
            let word = packed_weight[
                params.packed_weight_offset +
                first_column * params.words_per_row + word_in_row
            ];
            accumulator.x +=
                dot(activation, unpack4x8snorm(word)) * row_scales.x;
        }
        if (first_column + 1u < params.columns) {
            let word = packed_weight[
                params.packed_weight_offset +
                (first_column + 1u) * params.words_per_row + word_in_row
            ];
            accumulator.y +=
                dot(activation, unpack4x8snorm(word)) * row_scales.y;
        }
        if (first_column + 2u < params.columns) {
            let word = packed_weight[
                params.packed_weight_offset +
                (first_column + 2u) * params.words_per_row + word_in_row
            ];
            accumulator.z +=
                dot(activation, unpack4x8snorm(word)) * row_scales.z;
        }
        if (first_column + 3u < params.columns) {
            let word = packed_weight[
                params.packed_weight_offset +
                (first_column + 3u) * params.words_per_row + word_in_row
            ];
            accumulator.w +=
                dot(activation, unpack4x8snorm(word)) * row_scales.w;
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
