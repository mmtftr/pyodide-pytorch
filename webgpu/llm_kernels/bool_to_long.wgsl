struct Params {
    length: u32,
    ndim: u32,
    source_offset: u32,
    dispatch_x: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    source_strides0: vec4<u32>,
    source_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn size_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.sizes0[dim]; }
    return params.sizes1[dim - 4u];
}

fn source_stride_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.source_strides0[dim]; }
    return params.source_strides1[dim - 4u];
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let group = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let linear_index = group * 64u + local_id.x;
    if (linear_index >= params.length) { return; }

    var remaining = linear_index;
    var source_index = params.source_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % size_at(dim);
        remaining /= size_at(dim);
        source_index += coordinate * source_stride_at(dim);
    }

    // ATen Bool occupies one byte. WGSL storage accesses are word-sized, so
    // select the view's byte and canonicalize every nonzero representation.
    let source_word = source[source_index >> 2u];
    let source_shift = (source_index & 3u) * 8u;
    let source_byte = (source_word >> source_shift) & 0xffu;
    let value = select(0u, 1u, source_byte != 0u);

    // Restricted Long still uses ordinary eight-byte ATen storage. Bool
    // conversion is nonnegative, therefore its canonical high word is zero.
    let destination_word = linear_index * 2u;
    destination[destination_word] = value;
    destination[destination_word + 1u] = 0u;
}
