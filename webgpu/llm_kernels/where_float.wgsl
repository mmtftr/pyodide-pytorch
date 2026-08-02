const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    length: u32,
    ndim: u32,
    condition_ndim: u32,
    lhs_ndim: u32,
    rhs_ndim: u32,
    condition_offset: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    lhs_is_scalar: u32,
    rhs_is_scalar: u32,
    lhs_scalar_bits: u32,
    rhs_scalar_bits: u32,
    output_offset: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
    output_sizes: array<u32, MAX_DIMS>,
    condition_sizes: array<u32, MAX_DIMS>,
    condition_strides: array<u32, MAX_DIMS>,
    lhs_sizes: array<u32, MAX_DIMS>,
    lhs_strides: array<u32, MAX_DIMS>,
    rhs_sizes: array<u32, MAX_DIMS>,
    rhs_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read> condition: array<u32>;
@group(0) @binding(1) var<storage, read> lhs: array<f32>;
@group(0) @binding(2) var<storage, read> rhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn condition_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.condition_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.output_sizes[dim];
        remaining /= params.output_sizes[dim];
        if (dim + params.condition_ndim >= params.ndim) {
            let input_dim = dim + params.condition_ndim - params.ndim;
            if (params.condition_sizes[input_dim] != 1u) {
                index += coordinate * params.condition_strides[input_dim];
            }
        }
    }
    return index;
}

fn lhs_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.lhs_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.output_sizes[dim];
        remaining /= params.output_sizes[dim];
        if (dim + params.lhs_ndim >= params.ndim) {
            let input_dim = dim + params.lhs_ndim - params.ndim;
            if (params.lhs_sizes[input_dim] != 1u) {
                index += coordinate * params.lhs_strides[input_dim];
            }
        }
    }
    return index;
}

fn rhs_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.rhs_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.output_sizes[dim];
        remaining /= params.output_sizes[dim];
        if (dim + params.rhs_ndim >= params.ndim) {
            let input_dim = dim + params.rhs_ndim - params.ndim;
            if (params.rhs_sizes[input_dim] != 1u) {
                index += coordinate * params.rhs_strides[input_dim];
            }
        }
    }
    return index;
}

fn condition_at(index: u32) -> bool {
    let word = condition[index >> 2u];
    return ((word >> ((index & 3u) * 8u)) & 0xffu) != 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_index = gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    if (linear_index >= params.length) { return; }

    var left = bitcast<f32>(params.lhs_scalar_bits);
    if (params.lhs_is_scalar == 0u) {
        left = lhs[lhs_index(linear_index)];
    }
    var right = bitcast<f32>(params.rhs_scalar_bits);
    if (params.rhs_is_scalar == 0u) {
        right = rhs[rhs_index(linear_index)];
    }
    output[params.output_offset + linear_index] = select(
        right,
        left,
        condition_at(condition_index(linear_index)),
    );
}
