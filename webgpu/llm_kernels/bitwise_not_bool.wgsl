struct Params {
    length: u32,
    ndim: u32,
    input_offset: u32,
    dispatch_x: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    input_strides0: vec4<u32>,
    input_strides1: vec4<u32>,
};

// ATen Bool retains one byte per logical element. WGSL exposes storage through
// u32 words, so one invocation owns all four destination bytes in its word.
@group(0) @binding(0) var<storage, read> input: array<u32>;
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

fn bool_at(index: u32) -> bool {
    let word = input[index >> 2u];
    let shift = (index & 3u) * 8u;
    return ((word >> shift) & 0xffu) != 0u;
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
                !bool_at(input_index(linear_index))) {
            packed |= 1u << (byte * 8u);
        }
    }
    // Logical results and unused tail bytes are canonical one-byte Bool.
    output[output_word] = packed;
}
