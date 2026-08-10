struct Params {
    batch: u32,
    rows: u32,
    columns: u32,
    inner: u32,
    self_ndim: u32,
    self_offset: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    output_offset: u32,
    alpha_bits: u32,
    beta_bits: u32,
    _pad0: u32,
    self_sizes: array<u32, 4>,
    self_strides: array<u32, 4>,
    lhs_strides: array<u32, 4>,
    rhs_strides: array<u32, 4>,
};

@group(0) @binding(0) var<storage, read> self_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> lhs: array<f32>;
@group(0) @binding(2) var<storage, read> rhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn self_index(batch: u32, row: u32, column: u32) -> u32 {
    let coordinates = array<u32, 3>(batch, row, column);
    var index = params.self_offset;
    for (var output_dim = 0u; output_dim < 3u; output_dim++) {
        if (output_dim + params.self_ndim >= 3u) {
            let input_dim = output_dim + params.self_ndim - 3u;
            if (params.self_sizes[input_dim] != 1u) {
                index += coordinates[output_dim] * params.self_strides[input_dim];
            }
        }
    }
    return index;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let column = gid.x;
    let row = gid.y;
    let batch = gid.z;
    if (batch >= params.batch || row >= params.rows ||
            column >= params.columns) {
        return;
    }

    var dot = 0.0;
    for (var inner = 0u; inner < params.inner; inner++) {
        let lhs_index = params.lhs_offset +
            batch * params.lhs_strides[0] +
            row * params.lhs_strides[1] +
            inner * params.lhs_strides[2];
        let rhs_index = params.rhs_offset +
            batch * params.rhs_strides[0] +
            inner * params.rhs_strides[1] +
            column * params.rhs_strides[2];
        dot += lhs[lhs_index] * rhs[rhs_index];
    }
    let output_index = params.output_offset +
        (batch * params.rows + row) * params.columns + column;
    let beta = bitcast<f32>(params.beta_bits);
    var value = bitcast<f32>(params.alpha_bits) * dot;
    // Match native baddbmm: beta == 0 ignores self completely, including
    // NaN/Inf values that must not leak through a floating-point 0 * self.
    if (beta != 0.0) {
        value += beta * self_tensor[self_index(batch, row, column)];
    }
    output[output_index] = value;
}
