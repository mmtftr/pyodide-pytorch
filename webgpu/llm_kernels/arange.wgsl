struct Params {
    length: u32,
    dispatch_x: u32,
    output_offset: u32,
    output_words: u32,
    start_bits: u32,
    step_bits: u32,
    is_float: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x + gid.y * params.dispatch_x * 64u;
    if (index >= params.length) { return; }

    var low_word: u32;
    var high_word = 0u;
    if (params.is_float != 0u) {
        let value = bitcast<f32>(params.start_bits) +
            f32(index) * bitcast<f32>(params.step_bits);
        low_word = bitcast<u32>(value);
    } else {
        // Unsigned arithmetic gives the exact low word of two's-complement
        // addition even when the intermediate signed product would overflow.
        low_word = params.start_bits + index * params.step_bits;
        if ((low_word & 0x80000000u) != 0u) {
            high_word = 0xffffffffu;
        }
    }

    let word_offset = (params.output_offset + index) * params.output_words;
    output[word_offset] = low_word;
    if (params.output_words == 2u) {
        output[word_offset + 1u] = high_word;
    }
}
