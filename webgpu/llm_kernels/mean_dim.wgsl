struct Params {
    output_length: u32,
    reduce_size: u32,
    ndim: u32,
    reduce_dim: u32,
    input_offset: u32,
    output_offset: u32,
    reduce_stride: u32,
    dispatch_x: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    input_strides0: vec4<u32>,
    input_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> partial_sum: array<f32, 256>;

fn size_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.sizes0[dim]; }
    return params.sizes1[dim - 4u];
}

fn input_stride_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.input_strides0[dim]; }
    return params.input_strides1[dim - 4u];
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let output_index = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let lane = local_id.x;
    if (output_index >= params.output_length) { return; }

    // A contiguous output index enumerates all input coordinates except the
    // reduced dimension. Mapping it here avoids materializing strided views.
    var remaining = output_index;
    var input_base = params.input_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        if (dim != params.reduce_dim) {
            let coordinate = remaining % size_at(dim);
            remaining /= size_at(dim);
            input_base += coordinate * input_stride_at(dim);
        }
    }

    var sum = 0.0;
    for (var index = lane; index < params.reduce_size; index += 256u) {
        sum += input[input_base + index * params.reduce_stride];
    }
    partial_sum[lane] = sum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial_sum[lane] += partial_sum[lane + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        // For an empty reduction, IEEE 0 / 0 produces the NaN required by
        // aten::mean. The input binding is a nonempty dummy in that case.
        output[params.output_offset + output_index] =
            partial_sum[0] / f32(params.reduce_size);
    }
}
