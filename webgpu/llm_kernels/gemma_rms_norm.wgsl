struct Params {
    rows: u32,
    width: u32,
    input_offset: u32,
    output_offset: u32,
    weight_offset: u32,
    dispatch_x: u32,
    epsilon: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> offset_weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partial_square_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let lane = local_id.x;
    if (row >= params.rows) { return; }
    let input_base = params.input_offset + row * params.width;

    var square_sum = 0.0;
    for (var feature = lane; feature < params.width; feature += 256u) {
        let value = input[input_base + feature];
        square_sum += value * value;
    }
    partial_square_sum[lane] = square_sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial_square_sum[lane] += partial_square_sum[lane + stride];
        }
        workgroupBarrier();
    }

    let rstd = inverseSqrt(
        partial_square_sum[0] / f32(params.width) + params.epsilon);
    let output_base = params.output_offset + row * params.width;
    for (var feature = lane; feature < params.width; feature += 256u) {
        output[output_base + feature] =
            input[input_base + feature] * rstd *
            (1.0 + offset_weight[params.weight_offset + feature]);
    }
}
