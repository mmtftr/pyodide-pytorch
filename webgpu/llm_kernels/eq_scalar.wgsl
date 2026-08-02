struct Params {
    length: u32,
    ndim: u32,
    input_offset: u32,
    input_kind: u32,
    scalar_low: u32,
    scalar_high: u32,
    dispatch_x: u32,
    _pad0: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    input_strides0: vec4<u32>,
    input_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> input: array<u32>;
// ATen Bool storage is one byte per element. WGSL storage buffers expose u32
// words, so each invocation owns and writes four canonical 0/1 bytes.
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn size_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.sizes0[dim]; }
    return params.sizes1[dim - 4u];
}

fn input_stride_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.input_strides0[dim]; }
    return params.input_strides1[dim - 4u];
}

fn input_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.input_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % size_at(dim);
        remaining /= size_at(dim);
        index += coordinate * input_stride_at(dim);
    }
    return index;
}

fn equals_scalar(index: u32) -> bool {
    if (params.input_kind == 0u) {
        return bitcast<f32>(input[index]) == bitcast<f32>(params.scalar_low);
    }
    if (params.input_kind == 1u) {
        return input[index] == params.scalar_low;
    }
    // Restricted Long has ordinary two-word int64 storage. Comparing both
    // words also rejects any non-canonical value instead of truncating it.
    let word = index * 2u;
    return input[word] == params.scalar_low &&
        input[word + 1u] == params.scalar_high;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let group = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let output_word = group * 64u + local_id.x;
    let first_index = output_word * 4u;
    if (first_index >= params.length) { return; }

    var packed = 0u;
    for (var byte = 0u; byte < 4u; byte++) {
        let linear_index = first_index + byte;
        if (linear_index < params.length &&
                equals_scalar(input_index(linear_index))) {
            packed |= 1u << (byte * 8u);
        }
    }
    output[output_word] = packed;
}
