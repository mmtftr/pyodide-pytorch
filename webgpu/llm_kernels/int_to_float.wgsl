struct Params {
    length: u32,
    ndim: u32,
    source_offset: u32,
    destination_offset: u32,
    source_word_width: u32,
    destination_word_width: u32,
    convert_to_float: u32,
    dispatch_x: u32,
    sizes0: vec4<u32>,
    sizes1: vec4<u32>,
    source_strides0: vec4<u32>,
    source_strides1: vec4<u32>,
    destination_strides0: vec4<u32>,
    destination_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> source: array<i32>;
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

fn destination_stride_at(dim: u32) -> u32 {
    if (dim < 4u) { return params.destination_strides0[dim]; }
    return params.destination_strides1[dim - 4u];
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let group = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let linear_index = group * 64u + local_id.x;

    if (params.source_word_width == 0u) {
        // Same-dtype Bool `_to_copy`. Serial byte updates keep arbitrary
        // strided destinations race-free while preserving one-byte ATen
        // storage. Bool dtype conversion is intentionally not supported.
        if (linear_index != 0u) { return; }
        for (var bool_index = 0u;
             bool_index < params.length;
             bool_index++) {
            var remaining = bool_index;
            var source_index = params.source_offset;
            var destination_index = params.destination_offset;
            for (var reverse_dim = 0u;
                 reverse_dim < params.ndim;
                 reverse_dim++) {
                let dim = params.ndim - reverse_dim - 1u;
                let coordinate = remaining % size_at(dim);
                remaining /= size_at(dim);
                source_index += coordinate * source_stride_at(dim);
                destination_index +=
                    coordinate * destination_stride_at(dim);
            }
            let source_word = bitcast<u32>(source[source_index >> 2u]);
            let source_shift = (source_index & 3u) * 8u;
            let value = (source_word >> source_shift) & 0xffu;
            let destination_word = destination_index >> 2u;
            let destination_shift = (destination_index & 3u) * 8u;
            destination[destination_word] =
                (destination[destination_word] &
                    ~(0xffu << destination_shift)) |
                (value << destination_shift);
        }
        return;
    }

    if (linear_index >= params.length) { return; }

    var remaining = linear_index;
    var source_index = params.source_offset;
    var destination_index = params.destination_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % size_at(dim);
        remaining /= size_at(dim);
        source_index += coordinate * source_stride_at(dim);
        destination_index += coordinate * destination_stride_at(dim);
    }

    // Int uses one i32 word. Long is intentionally limited by the browser
    // profile to canonical sign-extended int32 values stored as two words, so
    // its low word is the exact signed value to convert.
    let source_word = source_index * params.source_word_width;
    let destination_word = destination_index * params.destination_word_width;
    if (params.convert_to_float != 0u) {
        if (params.convert_to_float == 2u) {
            let value = bitcast<f32>(source[source_word]);
            // Branch before conversion: converting NaN, infinity, or a value
            // outside i32 is not defined by the restricted-Long contract.
            if (!(value >= -2147483648.0 && value < 2147483648.0)) {
                destination[destination_word] = 0u;
                destination[destination_word + 1u] = 0x7fffffffu;
                return;
            }
            let converted = i32(trunc(value));
            destination[destination_word] = bitcast<u32>(converted);
            destination[destination_word + 1u] =
                select(0u, 0xffffffffu, converted < 0);
            return;
        }
        let value = source[source_word];
        if (params.source_word_width == 2u) {
            let expected_high = select(0, -1, value < 0);
            if (source[source_word + 1u] != expected_high) {
                destination[destination_word] = 0x7fc00000u;
                return;
            }
        }
        destination[destination_word] = bitcast<u32>(f32(value));
    } else {
        // `_to_copy` is a real copy even without a dtype change. Copy as raw
        // words so float NaN payloads and both words of limited Long survive.
        for (var word = 0u; word < params.source_word_width; word++) {
            destination[destination_word + word] =
                bitcast<u32>(source[source_word + word]);
        }
    }
}
