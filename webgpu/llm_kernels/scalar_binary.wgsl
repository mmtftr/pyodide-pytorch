const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    length: u32,
    ndim: u32,
    input_offset: u32,
    output_offset: u32,

    scalar_bits: u32,
    alpha_bits: u32,
    operation: u32,
    dispatch_x: u32,

    sizes: array<u32, MAX_DIMS>,
    input_strides: array<u32, MAX_DIMS>,
    output_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_index =
        gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    if (linear_index >= params.length) { return; }

    var remaining = linear_index;
    var input_index = params.input_offset;
    var output_index = params.output_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.sizes[dim];
        remaining /= params.sizes[dim];
        input_index += coordinate * params.input_strides[dim];
        output_index += coordinate * params.output_strides[dim];
    }

    let lhs = input[input_index];
    let rhs = bitcast<f32>(params.scalar_bits);
    let alpha = bitcast<f32>(params.alpha_bits);
    var value = lhs + alpha * rhs;
    if (params.operation == 1u) {
        value = lhs - alpha * rhs;
    } else if (params.operation == 2u) {
        value = lhs * rhs;
    } else if (params.operation == 3u) {
        value = lhs / rhs;
    }
    output[output_index] = value;
}
