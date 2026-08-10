const MAX_DIMS: u32 = 8u;
const WORKGROUP_SIZE: u32 = 64u;

const ADD: u32 = 0u;
const SUBTRACT: u32 = 1u;
const MULTIPLY: u32 = 2u;
const MINIMUM: u32 = 3u;
const REVERSE_SUBTRACT: u32 = 4u;
const NEGATE: u32 = 5u;
const ABSOLUTE: u32 = 6u;

struct Params {
    length: u32,
    ndim: u32,
    lhs_ndim: u32,
    rhs_ndim: u32,
    lhs_offset: u32,
    rhs_offset: u32,
    operation: u32,
    alpha_low: u32,
    scalar_low: u32,
    scalar_high: u32,
    rhs_is_scalar: u32,
    dispatch_x: u32,
    output_sizes: array<u32, MAX_DIMS>,
    lhs_sizes: array<u32, MAX_DIMS>,
    lhs_strides: array<u32, MAX_DIMS>,
    rhs_sizes: array<u32, MAX_DIMS>,
    rhs_strides: array<u32, MAX_DIMS>,
};

struct I64 {
    low: u32,
    high: u32,
};

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn canonical_high(low: u32) -> u32 {
    return select(0u, 0xffffffffu, (low & 0x80000000u) != 0u);
}

fn canonical(value: I64) -> bool {
    return value.high == canonical_high(value.low);
}

fn invalid() -> I64 {
    // Deliberately noncanonical: restricted-Long consumers contain it rather
    // than silently interpreting an out-of-profile value as signed int32.
    return I64(0u, 0x7fffffffu);
}

fn add64(left: I64, right: I64) -> I64 {
    let low = left.low + right.low;
    let carry = select(0u, 1u, low < left.low);
    return I64(low, left.high + right.high + carry);
}

fn negate64(value: I64) -> I64 {
    let low = ~value.low + 1u;
    let carry = select(0u, 1u, low == 0u);
    return I64(low, ~value.high + carry);
}

fn subtract64(left: I64, right: I64) -> I64 {
    return add64(left, negate64(right));
}

fn magnitude32(value: u32) -> u32 {
    return select(value, ~value + 1u, (value & 0x80000000u) != 0u);
}

fn unsigned_multiply32(left: u32, right: u32) -> I64 {
    let left0 = left & 0xffffu;
    let left1 = left >> 16u;
    let right0 = right & 0xffffu;
    let right1 = right >> 16u;
    let product0 = left0 * right0;
    let product1 = left0 * right1;
    let product2 = left1 * right0;
    let product3 = left1 * right1;

    var low = product0;
    var high = product3 + (product1 >> 16u) + (product2 >> 16u);
    let add1 = product1 << 16u;
    let previous1 = low;
    low += add1;
    high += select(0u, 1u, low < previous1);
    let add2 = product2 << 16u;
    let previous2 = low;
    low += add2;
    high += select(0u, 1u, low < previous2);
    return I64(low, high);
}

fn multiply_canonical(left: I64, right: I64) -> I64 {
    if (!canonical(left) || !canonical(right)) { return invalid(); }
    var result = unsigned_multiply32(
        magnitude32(left.low), magnitude32(right.low));
    let negative = ((left.low ^ right.low) & 0x80000000u) != 0u;
    if (negative) { result = negate64(result); }
    return result;
}

fn signed_less_canonical(left: I64, right: I64) -> bool {
    if (!canonical(left) || !canonical(right)) { return false; }
    return bitcast<i32>(left.low) < bitcast<i32>(right.low);
}

fn lhs_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.lhs_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.output_sizes[dim];
        remaining /= params.output_sizes[dim];
        if (dim + params.lhs_ndim >= params.ndim) {
            let input_dim = dim + params.lhs_ndim - params.ndim;
            if (params.lhs_sizes[input_dim] != 1u) {
                index += coordinate * params.lhs_strides[input_dim];
            }
        }
    }
    return index;
}

fn rhs_index(linear_index: u32) -> u32 {
    var remaining = linear_index;
    var index = params.rhs_offset;
    for (var reverse_dim = 0u; reverse_dim < params.ndim; reverse_dim++) {
        let dim = params.ndim - reverse_dim - 1u;
        let coordinate = remaining % params.output_sizes[dim];
        remaining /= params.output_sizes[dim];
        if (dim + params.rhs_ndim >= params.ndim) {
            let input_dim = dim + params.rhs_ndim - params.ndim;
            if (params.rhs_sizes[input_dim] != 1u) {
                index += coordinate * params.rhs_strides[input_dim];
            }
        }
    }
    return index;
}

fn load64(buffer: ptr<storage, array<u32>, read>, index: u32) -> I64 {
    return I64((*buffer)[index * 2u], (*buffer)[index * 2u + 1u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear_index = gid.x + gid.y * params.dispatch_x * WORKGROUP_SIZE;
    if (linear_index >= params.length) { return; }

    let left = load64(&lhs, lhs_index(linear_index));
    var right = I64(params.scalar_low, params.scalar_high);
    if (params.rhs_is_scalar == 0u) {
        right = load64(&rhs, rhs_index(linear_index));
    }

    let alpha = I64(params.alpha_low, canonical_high(params.alpha_low));
    var result = invalid();
    if (params.operation == ADD) {
        result = add64(left, multiply_canonical(right, alpha));
    } else if (params.operation == SUBTRACT) {
        result = subtract64(left, multiply_canonical(right, alpha));
    } else if (params.operation == MULTIPLY) {
        result = multiply_canonical(left, right);
    } else if (params.operation == MINIMUM) {
        if (canonical(left) && canonical(right)) {
            result = right;
            if (signed_less_canonical(left, right)) { result = left; }
        }
    } else if (params.operation == REVERSE_SUBTRACT) {
        result = subtract64(right, multiply_canonical(left, alpha));
    } else if (params.operation == NEGATE) {
        result = negate64(left);
    } else if (params.operation == ABSOLUTE) {
        result = left;
        if ((left.high & 0x80000000u) != 0u) { result = negate64(left); }
    }

    // The backend's Long profile contains only signed-int32-valued int64s.
    // Preserve that invariant for every successful producer; arithmetic that
    // leaves the profile becomes the deliberate noncanonical sentinel.
    if (!canonical(result)) { result = invalid(); }

    output[linear_index * 2u] = result.low;
    output[linear_index * 2u + 1u] = result.high;
}
