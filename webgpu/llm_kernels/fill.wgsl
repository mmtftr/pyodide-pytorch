struct Params {
    length: u32,
    dispatch_x: u32,
    output_offset: u32,
    output_words: u32,
    low_word: u32,
    high_word: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x + gid.y * params.dispatch_x * 64u;
    if (index >= params.length) { return; }

    if (params.output_words == 0u) {
        let range_start = params.output_offset;
        let range_end = range_start + params.high_word;
        let word_index = (range_start >> 2u) + index;
        let first_byte = word_index * 4u;
        var packed = output[word_index];
        for (var byte = 0u; byte < 4u; byte++) {
            let byte_index = first_byte + byte;
            if (byte_index >= range_start && byte_index < range_end) {
                let shift = byte * 8u;
                packed = (packed & ~(0xffu << shift)) |
                    ((params.low_word & 0xffu) << shift);
            }
        }
        output[word_index] = packed;
        return;
    }

    let word_offset = (params.output_offset + index) * params.output_words;
    output[word_offset] = params.low_word;
    if (params.output_words == 2u) {
        output[word_offset + 1u] = params.high_word;
    }
}
