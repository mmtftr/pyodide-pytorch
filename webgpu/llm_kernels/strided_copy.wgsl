struct Params {
    length: u32,
    ndim: u32,
    src_offset: u32,
    dst_offset: u32,
    element_words: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    src_strides0: vec4<u32>,
    src_strides1: vec4<u32>,
    dst_strides0: vec4<u32>,
    dst_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn size_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.sizes0[dim]; }
    return params.sizes1[dim - 4u];
}
fn source_stride_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.src_strides0[dim]; }
    return params.src_strides1[dim - 4u];
}

fn destination_stride_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.dst_strides0[dim]; }
    return params.dst_strides1[dim - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.element_words == 0u) {
        // Bool has one byte per ATen element, packed four-per-WGSL-word.
        // A single invocation serializes arbitrary strided destinations so
        // two elements can never race while updating bytes of the same word.
        if (gid.x != 0u) { return; }
        for (var linear_index = 0u;
             linear_index < params.length;
             linear_index++) {
            var remaining = linear_index;
            var src_index = params.src_offset;
            var dst_index = params.dst_offset;
            for (var reverse_dim = 0u;
                 reverse_dim < params.ndim;
                 reverse_dim++) {
                let dim = params.ndim - reverse_dim - 1u;
                let coordinate = remaining % size_at(dim);
                remaining /= size_at(dim);
                src_index += coordinate * source_stride_at(dim);
                dst_index += coordinate * destination_stride_at(dim);
            }
            let src_shift = (src_index & 3u) * 8u;
            let value = (source[src_index >> 2u] >> src_shift) & 0xffu;
            let dst_word = dst_index >> 2u;
            let dst_shift = (dst_index & 3u) * 8u;
            destination[dst_word] =
                (destination[dst_word] & ~(0xffu << dst_shift)) |
                (value << dst_shift);
        }
        return;
    }

    let linear_index = gid.x;
    if (linear_index >= params.length) { return; }

    var remaining = linear_index;
    var src_index = params.src_offset;
    var dst_index = params.dst_offset;
    for (var reverse_dim: u32 = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % size_at(dim);
        remaining /= size_at(dim);
        src_index += coordinate * source_stride_at(dim);
        dst_index += coordinate * destination_stride_at(dim);
    }
    let src_word = src_index * params.element_words;
    let dst_word = dst_index * params.element_words;
    for (var word = 0u; word < params.element_words; word++) {
        destination[dst_word + word] = source[src_word + word];
    }
}
