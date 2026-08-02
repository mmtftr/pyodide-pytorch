struct Params {
    elements: u32,
    batch: u32,
    heads: u32,
    tokens: u32,
    head_dim: u32,
    capacity: u32,
    key_state_offset: u32,
    value_state_offset: u32,
    key_cache_offset: u32,
    value_cache_offset: u32,
    position_offset: u32,
    position_words: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    key_state_strides: vec4<u32>,
    value_state_strides: vec4<u32>,
    key_cache_strides: vec4<u32>,
    value_cache_strides: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> key_states: array<f32>;
@group(0) @binding(1) var<storage, read> value_states: array<f32>;
@group(0) @binding(2) var<storage, read> cache_positions: array<u32>;
@group(0) @binding(3) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// Fused indexed writes avoid the four strided-copy dispatches and O(prefix)
// traffic incurred by two DynamicCache concatenations. Positions are signed
// int32 values stored either directly or in canonical sign-extended int64.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x + gid.y * params.dispatch_x * 64u;
    if (linear >= params.elements) { return; }

    var remaining = linear;
    let feature = remaining % params.head_dim;
    remaining /= params.head_dim;
    let token_index = remaining % params.tokens;
    remaining /= params.tokens;
    let head = remaining % params.heads;
    let batch_index = remaining / params.heads;

    let position_word =
        (params.position_offset + token_index) * params.position_words;
    let signed_position = bitcast<i32>(cache_positions[position_word]);
    var canonical = true;
    if (params.position_words == 2u) {
        let expected_high = select(
            0u,
            0xffffffffu,
            signed_position < 0,
        );
        canonical = cache_positions[position_word + 1u] == expected_high;
    }
    if (!canonical || signed_position < 0 ||
        u32(signed_position) >= params.capacity) {
        return;
    }
    let position = u32(signed_position);

    let key_source = params.key_state_offset +
        batch_index * params.key_state_strides[0] +
        head * params.key_state_strides[1] +
        token_index * params.key_state_strides[2] +
        feature * params.key_state_strides[3];
    let value_source = params.value_state_offset +
        batch_index * params.value_state_strides[0] +
        head * params.value_state_strides[1] +
        token_index * params.value_state_strides[2] +
        feature * params.value_state_strides[3];
    let key_destination = params.key_cache_offset +
        batch_index * params.key_cache_strides[0] +
        head * params.key_cache_strides[1] +
        position * params.key_cache_strides[2] +
        feature * params.key_cache_strides[3];
    let value_destination = params.value_cache_offset +
        batch_index * params.value_cache_strides[0] +
        head * params.value_cache_strides[1] +
        position * params.value_cache_strides[2] +
        feature * params.value_cache_strides[3];

    key_cache[key_destination] = key_states[key_source];
    value_cache[value_destination] = value_states[value_source];
}
