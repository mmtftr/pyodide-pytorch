const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    length: u32,
    ndim: u32,
    input_offset: u32,
    output_offset: u32,
    operation: u32,
    output_kind: u32,
    dispatch_x: u32,
    diagonal: i32,
    sizes: array<u32, MAX_DIMS>,
    input_strides: array<u32, MAX_DIMS>,
};

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn input_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.input_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.sizes[dim];
        remaining /= params.sizes[dim];
        index += coordinate * params.input_strides[dim];
    }
    return index;
}

fn keep(linear_index: u32) -> bool {
    let columns = params.sizes[params.ndim - 1u];
    let rows = params.sizes[params.ndim - 2u];
    let column = linear_index % columns;
    let row = (linear_index / columns) % rows;
    let relative = i32(column) - i32(row);
    if (params.operation == 0u) {
        return relative >= params.diagonal;
    }
    return relative <= params.diagonal;
}

fn input_bool(index: u32) -> bool {
    let word = input[index >> 2u];
    return ((word >> ((index & 3u) * 8u)) & 0xffu) != 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let unit = gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    if (params.output_kind == 0u) {
        if (unit >= params.length) { return; }
        output[params.output_offset + unit] = select(
            0u,
            input[input_index(unit)],
            keep(unit),
        );
        return;
    }

    // Bool output is contiguous and byte offset zero. A single invocation
    // owns the complete destination word, including canonical zero tail bytes.
    let first_index = unit * 4u;
    if (first_index >= params.length) { return; }
    var packed = 0u;
    for (var byte = 0u; byte < 4u; byte++) {
        let linear_index = first_index + byte;
        if (linear_index < params.length && keep(linear_index) &&
                input_bool(input_index(linear_index))) {
            packed |= 1u << (byte * 8u);
        }
    }
    output[unit] = packed;
}
