struct Params {
    length: u32,
    ndim: u32,
    input_offset: u32,
    _pad0: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    input_strides0: vec4<u32>,
    input_strides1: vec4<u32>,
};

// ATen Bool has one byte of storage per element; WGSL exposes those bytes four
// at a time through u32 storage words.
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial: array<u32, 256>;

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

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var value = 0u;
    for (var linear_index = lane;
         linear_index < params.length;
         linear_index += 256u) {
        if (bool_at(input_index(linear_index))) {
            value = 1u;
            break;
        }
    }
    partial[lane] = value;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] |= partial[lane + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        // The low byte is canonical true/false and the three padding bytes are
        // zero, matching a scalar ATen Bool allocation.
        output[0] = partial[0];
    }
}
