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

var<workgroup> partial_sum: array<f32, 256>;
var<workgroup> partial_square_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let lane = local_id.x;
    if (row >= params.rows) { return; }
    let base = params.input_offset + row * params.width;

    var sum = 0.0;
    var square_sum = 0.0;
    for (var feature = lane; feature < params.width; feature += 256u) {
        let value = input[base + feature];
        sum += value;
        square_sum += value * value;
    }
    partial_sum[lane] = sum;
    partial_square_sum[lane] = square_sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial_sum[lane] += partial_sum[lane + stride];
            partial_square_sum[lane] += partial_square_sum[lane + stride];
        }
        workgroupBarrier();
    }

    let mean = partial_sum[0] / f32(params.width);
    let variance = max(
        partial_square_sum[0] / f32(params.width) - mean * mean,
        0.0);
    let rstd = inverseSqrt(variance + params.epsilon);
    if (lane == 0u) {
        mean_output[params.mean_offset + row] = mean;
        rstd_output[params.rstd_offset + row] = rstd;
    }
    let output_base = params.output_offset + row * params.width;
    for (var feature = lane; feature < params.width; feature += 256u) {
        var value = (input[base + feature] - mean) * rstd;
        if (params.has_weight != 0u) {
            value *= weight[params.weight_offset + feature];
        }
        if (params.has_bias != 0u) {
            value += bias[params.bias_offset + feature];
        }
        output[output_base + feature] = value;
    }
}
