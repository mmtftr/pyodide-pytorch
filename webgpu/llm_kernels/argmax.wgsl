struct Params {
    output_length: u32,
    reduce_size: u32,
    ndim: u32,
    reduce_dim: u32,
    input_offset: u32,
    output_offset: u32,
    reduce_stride: u32,
    dispatch_x: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    input_strides0: vec4<u32>,
    input_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial_value: array<f32, 256>;
var<workgroup> partial_index: array<u32, 256>;

fn size_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.sizes0[dim]; }
    return params.sizes1[dim - 4u];
}

fn input_stride_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.input_strides0[dim]; }
    return params.input_strides1[dim - 4u];
}

fn should_replace(
    candidate_value: f32,
    candidate_index: u32,
    current_value: f32,
    current_index: u32) -> bool {
    if (candidate_index == 0xffffffffu) { return false; }
    if (current_index == 0xffffffffu) { return true; }

    // ATen treats NaN as greater than every numeric value for argmax. Ties,
    // including multiple NaNs, resolve to the lowest index.
    let candidate_nan = candidate_value != candidate_value;
    let current_nan = current_value != current_value;
    if (candidate_nan != current_nan) { return candidate_nan; }
    if (candidate_nan) { return candidate_index < current_index; }
    if (candidate_value > current_value) { return true; }
    if (candidate_value < current_value) { return false; }
    return candidate_index < current_index;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let output_index = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let lane = local_id.x;
    if (output_index >= params.output_length) { return; }

    var remaining = output_index;
    var input_base = params.input_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        if (dim != params.reduce_dim) {
            let coordinate = remaining % size_at(dim);
            remaining /= size_at(dim);
            input_base += coordinate * input_stride_at(dim);
        }
    }

    var best_index = 0xffffffffu;
    var best_value = 0.0;
    for (var index = lane; index < params.reduce_size; index += 256u) {
        let value = input[input_base + index * params.reduce_stride];
        if (should_replace(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    partial_value[lane] = best_value;
    partial_index[lane] = best_index;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            let candidate_value = partial_value[lane + stride];
            let candidate_index = partial_index[lane + stride];
            if (should_replace(
                    candidate_value,
                    candidate_index,
                    partial_value[lane],
                    partial_index[lane])) {
                partial_value[lane] = candidate_value;
                partial_index[lane] = candidate_index;
            }
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        // Restricted Long keeps ordinary 8-byte storage. A last-dimension
        // index is nonnegative and the C++ side bounds it to signed int32.
        let word_offset = (params.output_offset + output_index) * 2u;
        output[word_offset] = partial_index[0];
        output[word_offset + 1u] = 0u;
    }
}
