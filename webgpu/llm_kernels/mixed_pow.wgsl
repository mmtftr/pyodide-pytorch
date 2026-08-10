const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    length: u32,
    ndim: u32,
    base_ndim: u32,
    exponent_ndim: u32,
    base_offset: u32,
    exponent_offset: u32,
    exponent_words: u32,
    dispatch_x: u32,
    output_sizes: array<u32, MAX_DIMS>,
    base_sizes: array<u32, MAX_DIMS>,
    base_strides: array<u32, MAX_DIMS>,
    exponent_sizes: array<u32, MAX_DIMS>,
    exponent_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read> base: array<u32>;
@group(0) @binding(1) var<storage, read> exponent: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn base_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.base_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.output_sizes[dim];
        remaining /= params.output_sizes[dim];
        if (dim + params.base_ndim >= params.ndim) {
            let input_dim = dim + params.base_ndim - params.ndim;
            if (params.base_sizes[input_dim] != 1u) {
                index += coordinate * params.base_strides[input_dim];
            }
        }
    }
    return index;
}

fn exponent_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.exponent_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.output_sizes[dim];
        remaining /= params.output_sizes[dim];
        if (dim + params.exponent_ndim >= params.ndim) {
            let input_dim = dim + params.exponent_ndim - params.ndim;
            if (params.exponent_sizes[input_dim] != 1u) {
                index += coordinate * params.exponent_strides[input_dim];
            }
        }
    }
    return index;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_index = gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    if (linear_index >= params.length) { return; }
    let base_element = base_index(linear_index);
    let exponent_element = exponent_index(linear_index);
    let exponent_word = exponent_element * params.exponent_words;
    if (params.exponent_words == 0u) {
        output[linear_index] = pow(
            bitcast<f32>(base[base_element]),
            bitcast<f32>(exponent[exponent_element]),
        );
        return;
    }
    let exponent_low = exponent[exponent_word];
    if (params.exponent_words == 2u) {
        let expected_high = select(
            0u, 0xffffffffu, (exponent_low & 0x80000000u) != 0u);
        if (exponent[exponent_word + 1u] != expected_high) {
            output[linear_index] = bitcast<f32>(0x7fc00000u);
            return;
        }
    }
    output[linear_index] = pow(
        bitcast<f32>(base[base_element]),
        f32(bitcast<i32>(exponent_low)),
    );
}
