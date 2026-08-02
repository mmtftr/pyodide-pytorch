const MAX_DIMS: u32 = 8u;

struct Params {
    length: u32,
    ndim: u32,
    lhs_ndim: u32,
    rhs_ndim: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    dispatch_x: u32,
    _pad0: u32,
    output_sizes: array<u32, MAX_DIMS>,
    lhs_sizes: array<u32, MAX_DIMS>,
    lhs_strides: array<u32, MAX_DIMS>,
    rhs_sizes: array<u32, MAX_DIMS>,
    rhs_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
// ATen Bool uses one byte per logical element. Each invocation owns all four
// destination bytes in one WGSL word, preventing packed-byte write races.
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

fn bool_at(buffer: ptr<storage, array<u32>, read>, index: u32) -> bool {
    let word = (*buffer)[index >> 2u];
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
                bool_at(&lhs, lhs_index(linear_index)) &&
                bool_at(&rhs, rhs_index(linear_index))) {
            packed |= 1u << (byte * 8u);
        }
    }
    // Logical values and unused tail bytes are canonical one-byte Bool.
    output[output_word] = packed;
}
