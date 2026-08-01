const TILE: u32 = 8u;

struct Params {
    batch: u32,
    rows: u32,
    columns: u32,
    inner: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    output_offset: u32,
    _pad: u32,
    lhs_strides: vec4<u32>,
    rhs_strides: vec4<u32>,
    output_strides: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> lhs: array<f32>;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> lhs_tile: array<f32, 64>;
var<workgroup> rhs_tile: array<f32, 64>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch_index = workgroup_id.z;
    let row = workgroup_id.y * TILE + local_id.y;
    let column = workgroup_id.x * TILE + local_id.x;
    var accumulator = 0.0;

    let tile_count = (params.inner + TILE - 1u) / TILE;
    for (var tile: u32 = 0u; tile < tile_count; tile++) {
        let lhs_column = tile * TILE + local_id.x;
        let rhs_row = tile * TILE + local_id.y;
        let local_index = local_id.y * TILE + local_id.x;
        if (row < params.rows && lhs_column < params.inner) {
            let index = params.lhs_offset +
                batch_index * params.lhs_strides[0] +
                row * params.lhs_strides[1] +
                lhs_column * params.lhs_strides[2];
            lhs_tile[local_index] = lhs[index];
        } else {
            lhs_tile[local_index] = 0.0;
        }
        if (rhs_row < params.inner && column < params.columns) {
            let index = params.rhs_offset +
                batch_index * params.rhs_strides[0] +
                rhs_row * params.rhs_strides[1] +
                column * params.rhs_strides[2];
            rhs_tile[local_index] = rhs[index];
        } else {
            rhs_tile[local_index] = 0.0;
        }
        workgroupBarrier();
        for (var inner_index: u32 = 0u; inner_index < TILE; inner_index++) {
            accumulator += lhs_tile[local_id.y * TILE + inner_index] *
                rhs_tile[inner_index * TILE + local_id.x];
        }
        workgroupBarrier();
    }

    if (batch_index < params.batch && row < params.rows && column < params.columns) {
        let index = params.output_offset +
            batch_index * params.output_strides[0] +
            row * params.output_strides[1] +
            column * params.output_strides[2];
        output[index] = accumulator;
    }
}
