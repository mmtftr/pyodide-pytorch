const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    length: u32,
    ndim: u32,
    self_ndim: u32,
    mask_ndim: u32,
    self_offset: u32,
    mask_offset: u32,
    output_offset: u32,
    value_bits: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    output_sizes0: vec4<u32>,
    output_sizes1: vec4<u32>,
    self_sizes0: vec4<u32>,
    self_sizes1: vec4<u32>,
    self_strides0: vec4<u32>,
    self_strides1: vec4<u32>,
    mask_sizes0: vec4<u32>,
    mask_sizes1: vec4<u32>,
    mask_strides0: vec4<u32>,
    mask_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> self_values: array<f32>;
// ATen Bool storage is one byte per element. Reading a containing u32 is safe:
// this kernel never mutates the packed mask and each float output has exactly
// one invocation as its writer.
@group(0) @binding(1) var<storage, read> mask_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn split_at(first: vec4<u32>, second: vec4<u32>, dim: u32) -> u32 {
    if (dim < 4u) { return first[dim]; }
    return second[dim - 4u];
}

fn output_size(dim: u32) -> u32 {
    return split_at(params.output_sizes0, params.output_sizes1, dim);
}

fn self_size(dim: u32) -> u32 {
    return split_at(params.self_sizes0, params.self_sizes1, dim);
}

fn self_stride(dim: u32) -> u32 {
    return split_at(params.self_strides0, params.self_strides1, dim);
}

fn mask_size(dim: u32) -> u32 {
    return split_at(params.mask_sizes0, params.mask_sizes1, dim);
}

fn mask_stride(dim: u32) -> u32 {
    return split_at(params.mask_strides0, params.mask_strides1, dim);
}

fn self_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.self_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % output_size(dim);
        remaining /= output_size(dim);
        if (dim + params.self_ndim >= params.ndim) {
            let input_dim = dim + params.self_ndim - params.ndim;
            if (self_size(input_dim) != 1u) {
                index += coordinate * self_stride(input_dim);
            }
        }
    }
    return index;
}

fn mask_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.mask_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % output_size(dim);
        remaining /= output_size(dim);
        if (dim + params.mask_ndim >= params.ndim) {
            let input_dim = dim + params.mask_ndim - params.ndim;
            if (mask_size(input_dim) != 1u) {
                index += coordinate * mask_stride(input_dim);
            }
        }
    }
    return index;
}

fn bool_at(index: u32) -> bool {
    let word = mask_values[index >> 2u];
    return ((word >> ((index & 3u) * 8u)) & 0xffu) != 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_index = gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    if (linear_index >= params.length) { return; }

    output[params.output_offset + linear_index] = select(
        self_values[self_index(linear_index)],
        bitcast<f32>(params.value_bits),
        bool_at(mask_index(linear_index)),
    );
}
