// A finite sentinel keeps the online-maximum recurrence well-defined before
// the first unmasked score. Invalid lanes are assigned zero weight explicitly.
const NEGATIVE_SENTINEL: f32 = -3.402823e38;

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

// Each workgroup owns one [batch, query_head, query_position] row. A key tile
// has one key per invocation, so its score is calculated once and retained as
// its unnormalized softmax weight for the V pass.
var<workgroup> cached_query: array<f32, 256>;
var<workgroup> tile_weights: array<f32, 256>;
var<workgroup> reduction_maximum: array<f32, 256>;
var<workgroup> reduction_sum: array<f32, 256>;
var<workgroup> running_maximum: f32;
var<workgroup> running_sum: f32;
var<workgroup> rescale_previous: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let query_index = workgroup_id.x;
    let query_head = workgroup_id.y;
    let batch_index = workgroup_id.z;
    let lane = local_id.x;
    // The condition is workgroup-uniform because it only uses workgroup IDs.
    if (batch_index >= params.batch ||
        query_head >= params.query_heads ||
        query_index >= params.query_length) { return; }

    let heads_per_key_value_head =
        params.query_heads / params.key_value_heads;
    let key_value_head = query_head / heads_per_key_value_head;
    let query_base = params.query_offset +
        batch_index * params.query_strides[0] +
        query_head * params.query_strides[1] +
        query_index * params.query_strides[2];

    // Head dimensions through 256 (including the common D64/D128 cases) read
    // Q from storage only once. Larger dimensions retain correct behavior and
    // read their remaining features directly below.
    if (lane < params.head_dim) {
        cached_query[lane] =
            query[query_base + lane * params.query_strides[3]];
    } else {
        cached_query[lane] = 0.0;
    }
    if (lane == 0u) {
        running_maximum = NEGATIVE_SENTINEL;
        running_sum = 0.0;
        rescale_previous = 0.0;
    }
    workgroupBarrier();

    // PyTorch's is_causal contract is an upper-left lower triangle. Avoid
    // processing keys which cannot contribute to this query row.
    var active_key_length = params.key_value_length;
    if (params.causal != 0u) {
        active_key_length = min(active_key_length, query_index + 1u);
    }
    let tile_count = (active_key_length + 255u) / 256u;

    // Use all 256 invocations for the V pass at common head dimensions. D64
    // uses four key shards per feature and D65..D128 uses two; larger heads use
    // one invocation per feature in blocks of 256.
    var output_feature_tile = 256u;
    var output_key_shards = 1u;
    if (params.head_dim <= 64u) {
        output_feature_tile = 64u;
        output_key_shards = 4u;
    } else if (params.head_dim <= 128u) {
        output_feature_tile = 128u;
        output_key_shards = 2u;
    }
    let output_feature_lane = lane % output_feature_tile;
    let output_key_shard = lane / output_feature_tile;

    for (var tile: u32 = 0u; tile < tile_count; tile++) {
        let key_index = tile * 256u + lane;
        let has_key = key_index < active_key_length;
        var score = NEGATIVE_SENTINEL;
        if (has_key) {
            let key_base = params.key_offset +
                batch_index * params.key_strides[0] +
                key_value_head * params.key_strides[1] +
                key_index * params.key_strides[2];
            score = 0.0;
            let cached_features = min(params.head_dim, 256u);
            for (var feature: u32 = 0u;
                 feature < cached_features;
                 feature++) {
                score += cached_query[feature] *
                    key[key_base + feature * params.key_strides[3]];
            }
            for (var feature: u32 = 256u;
                 feature < params.head_dim;
                 feature++) {
                score += query[
                    query_base + feature * params.query_strides[3]] *
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
        }

        reduction_maximum[lane] = score;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                reduction_maximum[lane] = max(
                    reduction_maximum[lane],
                    reduction_maximum[lane + stride]);
            }
            workgroupBarrier();
        }

        if (lane == 0u) {
            let next_maximum = max(
                running_maximum, reduction_maximum[0]);
            rescale_previous = exp(running_maximum - next_maximum);
            running_maximum = next_maximum;
        }
        workgroupBarrier();

        var weight = 0.0;
        if (has_key) {
            weight = exp(score - running_maximum);
        }
        tile_weights[lane] = weight;
        reduction_sum[lane] = weight;
        workgroupBarrier();
        for (var stride = 128u; stride > 0u; stride >>= 1u) {
            if (lane < stride) {
                reduction_sum[lane] += reduction_sum[lane + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            running_sum = running_sum * rescale_previous + reduction_sum[0];
        }
        workgroupBarrier();

        for (var feature_base = 0u;
             feature_base < params.head_dim;
             feature_base += output_feature_tile) {
            let feature = feature_base + output_feature_lane;
            var tile_result = 0.0;
            if (feature < params.head_dim) {
                for (var tile_key = output_key_shard;
                     tile_key < 256u;
                     tile_key += output_key_shards) {
                    let current_key = tile * 256u + tile_key;
                    if (current_key < active_key_length) {
                        let weight_for_key = tile_weights[tile_key];
                        // This skips V traffic for causal, masked, or
                        // numerically underflowed zero-probability entries.
                        if (weight_for_key != 0.0) {
                            let value_index = params.value_offset +
                                batch_index * params.value_strides[0] +
                                key_value_head * params.value_strides[1] +
                                current_key * params.value_strides[2] +
                                feature * params.value_strides[3];
                            tile_result += weight_for_key * value[value_index];
                        }
                    }
                }
            }
            reduction_maximum[lane] = tile_result;
            workgroupBarrier();
            for (var shard_stride = output_key_shards >> 1u;
                 shard_stride > 0u;
                 shard_stride >>= 1u) {
                if (output_key_shard < shard_stride) {
                    reduction_maximum[lane] += reduction_maximum[
                        lane + shard_stride * output_feature_tile];
                }
                workgroupBarrier();
            }
            if (output_key_shard == 0u && feature < params.head_dim) {
                let output_index = params.output_offset +
                    batch_index * params.output_strides[0] +
                    query_head * params.output_strides[1] +
                    query_index * params.output_strides[2] +
                    feature * params.output_strides[3];
                var previous = 0.0;
                if (tile != 0u) {
                    previous = output[output_index];
                }
                output[output_index] = previous * rescale_previous +
                    reduction_maximum[lane];
            }
        }
    }

    // A zero denominator is the fully masked row convention used by SDPA.
    // Each output feature is finalized by the same invocation which owned its
    // online numerator, so no storage-memory barrier is required here.
    if (output_key_shard == 0u) {
        for (var feature = output_feature_lane;
             feature < params.head_dim;
             feature += output_feature_tile) {
            let output_index = params.output_offset +
                batch_index * params.output_strides[0] +
                query_head * params.output_strides[1] +
                query_index * params.output_strides[2] +
                feature * params.output_strides[3];
            if (running_sum == 0.0) {
                output[output_index] = 0.0;
            } else {
                output[output_index] = output[output_index] / running_sum;
            }
        }
    }
}
