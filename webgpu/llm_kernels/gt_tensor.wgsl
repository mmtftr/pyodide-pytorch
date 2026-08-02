const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    length: u32,
    ndim: u32,
    lhs_ndim: u32,
    rhs_ndim: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    input_kind: u32,
    dispatch_x: u32,
    output_sizes: array<u32, MAX_DIMS>,
    lhs_sizes: array<u32, MAX_DIMS>,
    lhs_strides: array<u32, MAX_DIMS>,
    rhs_sizes: array<u32, MAX_DIMS>,
    rhs_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
// ATen Bool storage is one byte per element. Each invocation owns all four
// output bytes in its word, so adjacent comparisons cannot race.
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

fn canonical_long(buffer: ptr<storage, array<u32>, read>, index: u32) -> bool {
    let low = bitcast<i32>((*buffer)[index * 2u]);
    let expected_high = select(0u, 0xffffffffu, low < 0);
    return (*buffer)[index * 2u + 1u] == expected_high;
}

fn greater(linear_index: u32) -> bool {
    let left_index = lhs_index(linear_index);
    let right_index = rhs_index(linear_index);
    if (params.input_kind == 0u) {
        return bitcast<f32>(lhs[left_index]) > bitcast<f32>(rhs[right_index]);
    }
    if (params.input_kind == 1u) {
        return bitcast<i32>(lhs[left_index]) > bitcast<i32>(rhs[right_index]);
    }
    // Restricted Long is physically int64 but every supported value is a
    // canonical sign extension of signed int32. Invalid encodings never
    // compare true; project-owned producers and uploads reject them earlier.
    return canonical_long(&lhs, left_index) && canonical_long(&rhs, right_index) &&
        bitcast<i32>(lhs[left_index * 2u]) >
            bitcast<i32>(rhs[right_index * 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_word = gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    let first_index = output_word * 4u;
    if (first_index >= params.length) { return; }

    var packed = 0u;
    for (var byte = 0u; byte < 4u; byte++) {
        let linear_index = first_index + byte;
        if (linear_index < params.length && greater(linear_index)) {
            packed |= 1u << (byte * 8u);
        }
    }
    output[output_word] = packed;
}
