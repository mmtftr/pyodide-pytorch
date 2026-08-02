struct Params {
    rows: u32,
    columns: u32,
    input_offset: u32,
    output_offset: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Restricted Long retains ordinary two-word, little-endian int64 storage.
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x + gid.y * params.dispatch_x * 64u;
    if (row >= params.rows) { return; }

    // Two-word addition avoids both f32 precision loss and silent int32
    // wrapping. Signed-int32-valued prefix sums therefore retain their exact
    // canonical 0/0xffffffff high word. A prefix outside that bounded profile
    // remains an exact int64 value and consequently has a noncanonical high
    // word, allowing restricted-Long consumers to contain it.
    var sum_low = 0u;
    var sum_high = 0u;
    for (var column = 0u; column < params.columns; column++) {
        let linear = row * params.columns + column;
        let input_word = (params.input_offset + linear) * 2u;
        let value_low = input[input_word];
        let value_high = input[input_word + 1u];
        let old_low = sum_low;
        sum_low += value_low;
        let carry = select(0u, 1u, sum_low < old_low);
        sum_high += value_high + carry;

        let output_word = (params.output_offset + linear) * 2u;
        output[output_word] = sum_low;
        output[output_word + 1u] = sum_high;
    }
}
