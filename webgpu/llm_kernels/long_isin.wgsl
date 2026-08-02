struct Params {
    length: u32,
    elements_ndim: u32,
    elements_offset: u32,
    test_length: u32,
    test_offset: u32,
    test_stride: u32,
    invert: u32,
    dispatch_x: u32,
    elements_sizes0: vec4<u32>,
    elements_sizes1: vec4<u32>,
    elements_strides0: vec4<u32>,
    elements_strides1: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> elements: array<u32>;
@group(0) @binding(1) var<storage, read> test_elements: array<u32>;
// ATen Bool uses one byte per element. One invocation owns all four bytes in
// its output word, avoiding byte-level read/modify/write races.
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn element_size(dim: u32) -> u32 {
    if (dim < 4u) { return params.elements_sizes0[dim]; }
    return params.elements_sizes1[dim - 4u];
}

fn element_stride(dim: u32) -> u32 {
    if (dim < 4u) { return params.elements_strides0[dim]; }
    return params.elements_strides1[dim - 4u];
}

fn element_storage_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var storage_index = params.elements_offset;
    for (var reverse_dim = 0u;
         reverse_dim < params.elements_ndim;
         reverse_dim++) {
        let dim = params.elements_ndim - reverse_dim - 1u;
        let coordinate = remaining % element_size(dim);
        remaining /= element_size(dim);
        storage_index += coordinate * element_stride(dim);
    }
    return storage_index;
}

fn canonical_high(low: u32) -> u32 {
    return select(0u, 0xffffffffu, (low & 0x80000000u) != 0u);
}

fn is_member(storage_index: u32) -> bool {
    let element_word = storage_index * 2u;
    let low = elements[element_word];
    let high = elements[element_word + 1u];
    // Out-of-profile int64 values never compare equal in this restricted
    // kernel, even if both tensors contain the same noncanonical bit pattern.
    if (high != canonical_high(low)) { return false; }

    for (var test_index = 0u;
         test_index < params.test_length;
         test_index++) {
        let test_storage_index =
            params.test_offset + test_index * params.test_stride;
        let test_word = test_storage_index * 2u;
        let test_low = test_elements[test_word];
        let test_high = test_elements[test_word + 1u];
        if (test_high == canonical_high(test_low) &&
                low == test_low && high == test_high) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {
    let group = workgroup_id.x + workgroup_id.y * params.dispatch_x;
    let output_word = group * 64u + local_id.x;
    let first_index = output_word * 4u;
    if (first_index >= params.length) { return; }

    var packed = 0u;
    for (var byte = 0u; byte < 4u; byte++) {
        let linear_index = first_index + byte;
        if (linear_index >= params.length) { continue; }
        var value = is_member(element_storage_index(linear_index));
        if (params.invert != 0u) { value = !value; }
        if (value) { packed |= 1u << (byte * 8u); }
    }
    output[output_word] = packed;
}
