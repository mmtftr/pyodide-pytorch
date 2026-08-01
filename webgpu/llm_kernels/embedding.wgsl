struct Params {
    num_indices: u32,
    embedding_dim: u32,
    num_embeddings: u32,
    weight_offset: u32,
    indices_offset: u32,
    output_offset: u32,
    weight_stride0: u32,
    weight_stride1: u32,
    dispatch_x: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_index = gid.x + gid.y * params.dispatch_x * 64u;
    let output_length = params.num_indices * params.embedding_dim;
    if (output_index >= output_length) { return; }

    let token_position = output_index / params.embedding_dim;
    let feature = output_index % params.embedding_dim;
    let token = indices[params.indices_offset + token_position];
    if (token < 0 || u32(token) >= params.num_embeddings) {
        output[params.output_offset + output_index] = 0.0;
        return;
    }
    let weight_index = params.weight_offset +
        u32(token) * params.weight_stride0 + feature * params.weight_stride1;
    output[params.output_offset + output_index] = weight[weight_index];
}
