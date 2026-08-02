struct Params {
    rows: u32,
    width: u32,
    input_offset: u32,
    output_offset: u32,
    mean_offset: u32,
    rstd_offset: u32,
    weight_offset: u32,
    bias_offset: u32,
    has_weight: u32,
    has_bias: u32,
    dispatch_x: u32,
    _pad: u32,
    epsilon: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> mean_output: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd_output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

// Each invocation accumulates a disjoint strided slice with Welford's online
// algorithm.  The tree below then combines those (count, mean, M2) states.
// This avoids the catastrophic cancellation in E[x^2] - E[x]^2 for hidden
// states with a large common offset.
var<workgroup> partial_count: array<u32, 256>;
var<workgroup> partial_mean: array<f32, 256>;
var<workgroup> partial_m2: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let lane = local_id.x;
    if (row >= params.rows) { return; }
    let base = params.input_offset + row * params.width;

    var count = 0u;
    var mean = 0.0;
    var m2 = 0.0;
    for (var feature = lane; feature < params.width; feature += 256u) {
        let value = input[base + feature];
        count += 1u;
        let delta = value - mean;
        mean += delta / f32(count);
        let adjusted_delta = value - mean;
        m2 += delta * adjusted_delta;
    }
    partial_count[lane] = count;
    partial_mean[lane] = mean;
    partial_m2[lane] = m2;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            let right_count = partial_count[lane + stride];
            if (right_count != 0u) {
                let left_count = partial_count[lane];
                if (left_count == 0u) {
                    partial_count[lane] = right_count;
                    partial_mean[lane] = partial_mean[lane + stride];
                    partial_m2[lane] = partial_m2[lane + stride];
                } else {
                    let combined_count = left_count + right_count;
                    let delta = partial_mean[lane + stride] - partial_mean[lane];
                    let left_fraction = f32(left_count) / f32(combined_count);
                    let right_fraction = f32(right_count) / f32(combined_count);
                    partial_mean[lane] += delta * right_fraction;
                    partial_m2[lane] += partial_m2[lane + stride] +
                        delta * delta * f32(combined_count) *
                        left_fraction * right_fraction;
                    partial_count[lane] = combined_count;
                }
            }
        }
        workgroupBarrier();
    }

    let row_mean = partial_mean[0];
    let variance = max(partial_m2[0] / f32(params.width), 0.0);
    let row_rstd = inverseSqrt(variance + params.epsilon);
    if (lane == 0u) {
        mean_output[params.mean_offset + row] = row_mean;
        rstd_output[params.rstd_offset + row] = row_rstd;
    }
    let output_base = params.output_offset + row * params.width;
    for (var feature = lane; feature < params.width; feature += 256u) {
        var value = (input[base + feature] - row_mean) * row_rstd;
        if (params.has_weight != 0u) {
            value *= weight[params.weight_offset + feature];
        }
        if (params.has_bias != 0u) {
            value += bias[params.bias_offset + feature];
        }
        output[output_base + feature] = value;
    }
}
