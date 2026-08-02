const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    length: u32,
    ndim: u32,
    rhs_ndim: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
    sizes: array<u32, MAX_DIMS>,
    lhs_strides: array<u32, MAX_DIMS>,
    rhs_sizes: array<u32, MAX_DIMS>,
    rhs_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read_write> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn indices(linear_index: u32) -> vec2<u32> {
    var remaining = linear_index;
    var lhs_index = params.lhs_offset;
    var rhs_index = params.rhs_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.sizes[dim];
        remaining /= params.sizes[dim];
        lhs_index += coordinate * params.lhs_strides[dim];
        if (dim + params.rhs_ndim >= params.ndim) {
            let input_dim = dim + params.rhs_ndim - params.ndim;
            if (params.rhs_sizes[input_dim] != 1u) {
                rhs_index += coordinate * params.rhs_strides[input_dim];
            }
        }
    }
    return vec2(lhs_index, rhs_index);
}

fn bool_at(index: u32) -> bool {
    let word = rhs[index >> 2u];
    return ((word >> ((index & 3u) * 8u)) & 0xffu) != 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_index = gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    if (linear_index >= params.length) { return; }
    let locations = indices(linear_index);
    lhs[locations.x] *= select(0.0, 1.0, bool_at(locations.y));
}
