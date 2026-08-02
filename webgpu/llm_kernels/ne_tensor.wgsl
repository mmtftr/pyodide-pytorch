const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    lhs_ndim: u32,
    rhs_ndim: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    input_kinds: u32,
    dispatch_x: u32,
    output_sizes: array<u32, MAX_DIMS>,
    lhs_sizes: array<u32, MAX_DIMS>,
    lhs_strides: array<u32, MAX_DIMS>,
    rhs_sizes: array<u32, MAX_DIMS>,
    rhs_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
// ATen Bool uses one byte per logical element. Each invocation owns a complete
// output word, including its unused tail bytes, so packed writes cannot race.
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

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

fn canonical_high(low: u32) -> u32 {
    return select(0u, 0xffffffffu, bitcast<i32>(low) < 0);
}

fn unequal(linear_index: u32) -> bool {
    let left_index = lhs_index(linear_index);
    let right_index = rhs_index(linear_index);
    let lhs_is_long = (params.input_kinds & 1u) != 0u;
    let rhs_is_long = (params.input_kinds & 2u) != 0u;
    let lhs_word = select(left_index, left_index * 2u, lhs_is_long);
    let rhs_word = select(right_index, right_index * 2u, rhs_is_long);
    let lhs_low = lhs[lhs_word];
    let rhs_low = rhs[rhs_word];
    if (lhs_low != rhs_low) { return true; }

    if (lhs_is_long && rhs_is_long) {
        return lhs[lhs_word + 1u] != rhs[rhs_word + 1u];
    }
    if (lhs_is_long) {
        return lhs[lhs_word + 1u] != canonical_high(lhs_low);
    }
    if (rhs_is_long) {
        return rhs[rhs_word + 1u] != canonical_high(rhs_low);
    }
    return false;
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
        if (linear_index < params.length && unequal(linear_index)) {
            packed |= 1u << (byte * 8u);
        }
    }
    output[output_word] = packed;
}
