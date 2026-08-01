const NEGATIVE_INFINITY: f32 = -3.402823e38;

struct Params {
    batch: u32,
    query_heads: u32,
    key_value_heads: u32,
    query_length: u32,
    key_value_length: u32,
    head_dim: u32,
    query_offset: u32,
    key_offset: u32,
    value_offset: u32,
    output_offset: u32,
    mask_offset: u32,
    has_mask: u32,
    causal: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    scale: f32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    query_strides: vec4<u32>,
    key_strides: vec4<u32>,
    value_strides: vec4<u32>,
    output_strides: vec4<u32>,
    mask_strides: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partial_maximum: array<f32, 256>;
var<workgroup> partial_sum: array<f32, 256>;

fn attention_score(
    batch_index: u32,
    query_head: u32,
    query_index: u32,
    key_index: u32) -> f32 {
    if (params.causal != 0u) {
        if (key_index > query_index) { return NEGATIVE_INFINITY; }
    }
    let heads_per_key_value_head =
        params.query_heads / params.key_value_heads;
    let key_value_head = query_head / heads_per_key_value_head;
    let query_base = params.query_offset +
        batch_index * params.query_strides[0] +
        query_head * params.query_strides[1] +
        query_index * params.query_strides[2];
    let key_base = params.key_offset +
        batch_index * params.key_strides[0] +
        key_value_head * params.key_strides[1] +
        key_index * params.key_strides[2];
    var score = 0.0;
    for (var feature: u32 = 0u; feature < params.head_dim; feature++) {
        score += query[query_base + feature * params.query_strides[3]] *
            key[key_base + feature * params.key_strides[3]];
    }
    score *= params.scale;
    if (params.has_mask != 0u) {
        let mask_index = params.mask_offset +
            batch_index * params.mask_strides[0] +
            query_head * params.mask_strides[1] +
            query_index * params.mask_strides[2] +
            key_index * params.mask_strides[3];
        score += mask[mask_index];
    }
    return score;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let query_index = workgroup_id.x;
    let query_head = workgroup_id.y;
    let batch_index = workgroup_id.z;
    let lane = local_id.x;
    if (batch_index >= params.batch ||
        query_head >= params.query_heads ||
        query_index >= params.query_length) { return; }

    var local_maximum = NEGATIVE_INFINITY;
    for (var key_index = lane;
         key_index < params.key_value_length;
         key_index += 256u) {
        local_maximum = max(local_maximum, attention_score(
            batch_index, query_head, query_index, key_index));
    }
    partial_maximum[lane] = local_maximum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial_maximum[lane] = max(
                partial_maximum[lane], partial_maximum[lane + stride]);
        }
        workgroupBarrier();
    }
    let maximum = partial_maximum[0];

    var local_sum = 0.0;
    for (var key_index = lane;
         key_index < params.key_value_length;
         key_index += 256u) {
        local_sum += exp(attention_score(
            batch_index, query_head, query_index, key_index) - maximum);
    }
    partial_sum[lane] = local_sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial_sum[lane] += partial_sum[lane + stride];
        }
        workgroupBarrier();
    }
    let denominator = partial_sum[0];
    let heads_per_key_value_head =
        params.query_heads / params.key_value_heads;
    let key_value_head = query_head / heads_per_key_value_head;

    for (var feature = lane; feature < params.head_dim; feature += 256u) {
        var result = 0.0;
        for (var key_index: u32 = 0u;
             key_index < params.key_value_length;
             key_index++) {
            let probability = exp(attention_score(
                batch_index, query_head, query_index, key_index) - maximum) /
                denominator;
            let value_index = params.value_offset +
                batch_index * params.value_strides[0] +
                key_value_head * params.value_strides[1] +
                key_index * params.value_strides[2] +
                feature * params.value_strides[3];
            result += probability * value[value_index];
        }
        let output_index = params.output_offset +
            batch_index * params.output_strides[0] +
            query_head * params.output_strides[1] +
            query_index * params.output_strides[2] +
            feature * params.output_strides[3];
        output[output_index] = result;
    }
}
