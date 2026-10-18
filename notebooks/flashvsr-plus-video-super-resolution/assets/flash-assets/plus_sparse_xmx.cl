// Selected-block BF16 attention, with the direct and split-key entries in one
// source.  The default path is the qualified B580 SIMD16/2D implementation.
// OTC_A770_SG8 selects the scalar-load SIMD8 path: the XML still has
// a 16-wide work group. Split partial reuses K/V across eight query rows in
// one subgroup; direct and merge use two four-row subgroups.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

#if !defined(OTC_A770_SG8) && !defined(cl_intel_subgroup_2d_block_io)
// DG2/A770 does not expose Intel's 2D block extension. Select the
// SG8 scalar path from the capability macro rather than requiring a device
// name or a second source file.  A caller may still define OTC_A770_SG8
// explicitly when compiling a capability probe.
#define OTC_A770_SG8 1
#endif

#if defined(OTC_A770_SG8)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#if !defined(OTC_A770_SG8)
#ifdef cl_intel_subgroup_2d_block_io
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
#else
#error "The SIMD16 path requires cl_intel_subgroup_2d_block_io"
#endif
#endif

#ifndef cl_intel_subgroup_matrix_multiply_accumulate
#error "BF16 DPAS is required"
#endif
#ifndef cl_intel_bfloat16_conversions
#error "BF16 conversion intrinsics are required"
#endif
#if defined(OTC_A770_SG8) && !defined(cl_intel_subgroups)
#error "Subgroup shuffle/reduction intrinsics are required"
#endif

#ifndef OTC_SPLIT_KEY
#define OTC_SPLIT_KEY 4
#endif
#if OTC_SPLIT_KEY != 2 && OTC_SPLIT_KEY != 4 && OTC_SPLIT_KEY != 8
#error "Only reviewed split counts 2, 4 and 8 are admitted"
#endif

#if defined(OTC_A770_SG8)
#if defined(OTC_SPLIT_PARTIAL)
#define OTC_ROWS 8
typedef int8 otc_a_t;
typedef float8 otc_f_t;
#else
#define OTC_ROWS 4
typedef int4 otc_a_t;
typedef float4 otc_f_t;
#endif
#define OTC_OUT_TILES 16
#define OTC_KEY_GROUPS 4
#define OTC_GROUP_SIZE 8
#else
#define OTC_ROWS 8
#define OTC_OUT_TILES 8
#define OTC_KEY_GROUPS 2
#define OTC_GROUP_SIZE 16
typedef short8 otc_a_t;
typedef float8 otc_f_t;
#endif
typedef int8 otc_b_t;

// Every raw pair is two adjacent BF16 words.  Keeping this helper in terms
// of words, rather than converting through half, is important for signed zero
// and subnormal values in the attention inputs.
inline uint otc_pair(__global const ushort *p, size_t base, uint row,
                     uint column, uint row_pitch) {
    const size_t at = base + (size_t)row * row_pitch + column;
    return (uint)p[at] | ((uint)p[at + 1] << 16);
}

#if defined(OTC_A770_SG8)
inline uint otc_vertical_pair(__global const ushort *p, size_t base, uint row,
                              uint column, uint row_pitch) {
    const size_t at = base + (size_t)row * row_pitch + column;
    return (uint)p[at] | ((uint)p[at + row_pitch] << 16);
}

inline otc_a_t otc_load_query(__global const ushort *q, size_t base,
                              uint tile, uint row_pitch, uint lane) {
    otc_a_t result;
    for (uint row = 0; row < OTC_ROWS; ++row)
        result[row] = as_int(otc_pair(q, base, row, tile * 16 + lane * 2,
                                      row_pitch));
    return result;
}

inline otc_b_t otc_load_key(__global const ushort *k, size_t base, uint start,
                            uint group, uint tile, uint row_pitch, uint lane) {
    otc_b_t result;
    for (uint i = 0; i < 8; ++i)
        result[i] = as_int(otc_pair(k, base, start + group * 8 + lane,
                                    tile * 16 + i * 2, row_pitch));
    return result;
}

inline otc_b_t otc_load_value(__global const ushort *v, size_t base, uint start,
                              uint group, uint tile, uint row_pitch, uint lane) {
    otc_b_t result;
    for (uint i = 0; i < 8; ++i)
        result[i] = as_int(otc_vertical_pair(
            v, base, start + group * 16 + i * 2, tile * 8 + lane, row_pitch));
    return result;
}

// The SG8 matrix A operand packs two adjacent key columns per lane.  All
// shuffles are unconditional: subgroup built-ins must not sit on the
// lane<4/lane>=4 branch that selects the two eight-column halves.
inline otc_a_t otc_pack_probability(otc_f_t first, otc_f_t second, uint lane) {
    otc_a_t result;
    const uint pair_lane = (lane & 3) * 2;
    for (uint row = 0; row < OTC_ROWS; ++row) {
        const uint a = (uint)intel_convert_bfloat16_as_ushort(first[row]);
        const uint b = (uint)intel_convert_bfloat16_as_ushort(second[row]);
        const uint a0 = intel_sub_group_shuffle(a, pair_lane);
        const uint a1 = intel_sub_group_shuffle(a, pair_lane + 1);
        const uint b0 = intel_sub_group_shuffle(b, pair_lane);
        const uint b1 = intel_sub_group_shuffle(b, pair_lane + 1);
        const uint lo = lane < 4 ? a0 : b0;
        const uint hi = lane < 4 ? a1 : b1;
        result[row] = as_int(lo | (hi << 16));
    }
    return result;
}
#else
inline otc_a_t otc_load_query(__global const ushort *q, size_t base,
                              uint tile, uint row_pitch, uint lane) {
    otc_a_t result;
    intel_sub_group_2d_block_read_16b_8r16x1c(
        (__global void *)(q + base), 256, 8, 256,
        (int2)(tile * 16, 0), (__private ushort *)&result);
    return result;
}

inline otc_b_t otc_load_key(__global const ushort *k, size_t base, uint start,
                            uint group, uint tile, uint row_pitch, uint lane) {
    otc_b_t result;
    intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
        (__global void *)(k + base), 256, 128, 2 * row_pitch,
        (int2)(tile * 8, start + group * 16), (__private uint *)&result);
    return result;
}

inline otc_b_t otc_load_value(__global const ushort *v, size_t base, uint start,
                              uint group, uint tile, uint row_pitch, uint lane) {
    otc_b_t result;
    intel_sub_group_2d_block_read_transform_16b_16r16x1c(
        (__global void *)(v + base), 256, 128, 2 * row_pitch,
        (int2)(tile * 16, start + group * 16), (__private uint *)&result);
    return result;
}
#endif

#if defined(OTC_CROSS_PV_SPLIT)
#define OTC_HAS_RESIDUAL 1
#endif

// One shared attention loop serves direct, precise-cross, and split partial
// entries.  In split mode the output is numerator/max/denominator scratch;
// otherwise it is the existing raw-BF16 output ABI.  Merge has no Q/K/V
// inputs, so leave this helper out of that translation unit entirely; the
// custom-layer JIT only defines metadata macros for actual ports.
#if !defined(OTC_SPLIT_MERGE)
static inline void otc_run_attention(
    __global const ushort *q, __global const ushort *k,
    __global const ushort *v, __global const int *blocks,
    __global void *result, uint head, uint window, uint first_query,
    uint begin, uint end, bool canonical_kv, uint key_windows,
    size_t qbase, size_t target, size_t selection, uint lane) {
    otc_a_t query[8];
    otc_f_t accumulator[OTC_OUT_TILES];
    const uint krow_pitch = canonical_kv ? INPUT1_PITCHES[1] : INPUT1_PITCHES[2];
    const uint vrow_pitch = canonical_kv ? INPUT2_PITCHES[1] : INPUT2_PITCHES[2];
    for (uint tile = 0; tile < 8; ++tile) {
        query[tile] = otc_load_query(q, qbase, tile, INPUT0_PITCHES[2], lane);
    }
    for (uint tile = 0; tile < OTC_OUT_TILES; ++tile)
        accumulator[tile] = (otc_f_t)(0);
    otc_f_t maximum = (otc_f_t)(-INFINITY);
    otc_f_t denominator = (otc_f_t)(0);
    for (uint slot = begin; slot < end; ++slot) {
        const int block = blocks[selection + slot * INPUT3_PITCHES[3]];
        if (block < 0 || (uint)block >= key_windows)
            continue;
        const size_t kbase = INPUT1_OFFSET + (canonical_kv
            ? (size_t)block * INPUT1_PITCHES[0] + head * INPUT1_PITCHES[2]
            : head * INPUT1_PITCHES[0] + (size_t)block * INPUT1_PITCHES[1]);
        const size_t vbase = INPUT2_OFFSET + (canonical_kv
            ? (size_t)block * INPUT2_PITCHES[0] + head * INPUT2_PITCHES[2]
            : head * INPUT2_PITCHES[0] + (size_t)block * INPUT2_PITCHES[1]);
        for (uint start = 0; start < 128; start += 32) {
            otc_f_t logits[OTC_KEY_GROUPS];
            for (uint group = 0; group < OTC_KEY_GROUPS; ++group)
                logits[group] = (otc_f_t)(0);
            for (uint depth = 0; depth < 8; ++depth) {
                for (uint group = 0; group < OTC_KEY_GROUPS; ++group)
                    logits[group] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                        query[depth], otc_load_key(k, kbase, start, group, depth,
                                                   krow_pitch, lane), logits[group]);
            }
            for (uint group = 0; group < OTC_KEY_GROUPS; ++group)
                logits[group] *= 0.08838834764831844f;

            otc_a_t probability[2];
#if defined(OTC_HAS_RESIDUAL)
            otc_a_t residual_probability[2];
#endif
            otc_f_t correction = (otc_f_t)(0);
            otc_f_t converted[OTC_KEY_GROUPS];
#if defined(OTC_A770_SG8)
            for (uint row = 0; row < OTC_ROWS; ++row) {
                float local_max = -INFINITY;
                for (uint group = 0; group < OTC_KEY_GROUPS; ++group)
                    local_max = fmax(local_max,
                        sub_group_reduce_max(logits[group][row]));
                const float updated = fmax(maximum[row], local_max);
                correction[row] = exp(maximum[row] - updated);
                for (uint group = 0; group < OTC_KEY_GROUPS; ++group)
                    converted[group][row] = exp(logits[group][row] - updated);
                float added = 0.0f;
                for (uint group = 0; group < OTC_KEY_GROUPS; ++group)
                    added += converted[group][row];
                denominator[row] = denominator[row] * correction[row]
                    + sub_group_reduce_add(added);
                maximum[row] = updated;
            }
            probability[0] = otc_pack_probability(converted[0], converted[1], lane);
            probability[1] = otc_pack_probability(converted[2], converted[3], lane);
#if defined(OTC_HAS_RESIDUAL)
            otc_f_t residual[OTC_KEY_GROUPS];
            for (uint group = 0; group < OTC_KEY_GROUPS; ++group)
                for (uint row = 0; row < OTC_ROWS; ++row) {
                    const uint bits = (uint)intel_convert_bfloat16_as_ushort(
                        converted[group][row]);
                    residual[group][row] = converted[group][row]
                        - as_float(bits << 16);
                }
            residual_probability[0] = otc_pack_probability(residual[0], residual[1], lane);
            residual_probability[1] = otc_pack_probability(residual[2], residual[3], lane);
#endif
#else
            for (uint row = 0; row < OTC_ROWS; ++row) {
                const float local_max = sub_group_reduce_max(
                    fmax(logits[0][row], logits[1][row]));
                const float updated = fmax(maximum[row], local_max);
                correction[row] = exp(maximum[row] - updated);
                const float p0 = exp(logits[0][row] - updated);
                const float p1 = exp(logits[1][row] - updated);
                denominator[row] = denominator[row] * correction[row]
                    + sub_group_reduce_add(p0 + p1);
                maximum[row] = updated;
                probability[0][row] = as_short(
                    intel_convert_bfloat16_as_ushort(p0));
                probability[1][row] = as_short(
                    intel_convert_bfloat16_as_ushort(p1));
#if defined(OTC_HAS_RESIDUAL)
                const ushort b0 = intel_convert_bfloat16_as_ushort(p0);
                const ushort b1 = intel_convert_bfloat16_as_ushort(p1);
                residual_probability[0][row] = as_short(
                    intel_convert_bfloat16_as_ushort(
                        p0 - as_float((uint)b0 << 16)));
                residual_probability[1][row] = as_short(
                    intel_convert_bfloat16_as_ushort(
                        p1 - as_float((uint)b1 << 16)));
#endif
            }
#endif

            for (uint depth = 0; depth < OTC_OUT_TILES; ++depth) {
                otc_f_t updated = accumulator[depth] * correction;
                const otc_b_t value0 = otc_load_value(
                    v, vbase, start, 0, depth, vrow_pitch, lane);
                const otc_b_t value1 = otc_load_value(
                    v, vbase, start, 1, depth, vrow_pitch, lane);
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    probability[0], value0, updated);
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    probability[1], value1, updated);
#if defined(OTC_HAS_RESIDUAL)
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    residual_probability[0], value0, updated);
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    residual_probability[1], value1, updated);
#endif
                accumulator[depth] = updated;
            }
        }
    }

#if defined(OTC_SPLIT_PARTIAL)
    __global float *partial = (__global float *)result;
    for (uint depth = 0; depth < OTC_OUT_TILES; ++depth)
        for (uint row = 0; row < OTC_ROWS; ++row)
            partial[target + row * OUTPUT0_PITCHES[2] +
                (depth * OTC_GROUP_SIZE + lane) * OUTPUT0_PITCHES[3]] =
                accumulator[depth][row];
    if (lane == 0)
        for (uint row = 0; row < OTC_ROWS; ++row) {
            partial[target + row * OUTPUT0_PITCHES[2] + 128 * OUTPUT0_PITCHES[3]] = maximum[row];
            partial[target + row * OUTPUT0_PITCHES[2] + 129 * OUTPUT0_PITCHES[3]] = denominator[row];
        }
#else
    __global ushort *output = (__global ushort *)result;
#if defined(OTC_A770_SG8)
    for (uint depth = 0; depth < OTC_OUT_TILES; ++depth)
        for (uint row = 0; row < OTC_ROWS; ++row)
            output[target + row * OUTPUT0_PITCHES[2] +
                (depth * OTC_GROUP_SIZE + lane) * OUTPUT0_PITCHES[3]] =
                denominator[row] == 0.0f ? (ushort)0 :
                    intel_convert_bfloat16_as_ushort(
                        accumulator[depth][row] / denominator[row]);
#else
    for (uint depth = 0; depth < OTC_OUT_TILES; ++depth) {
        ushort8 pixels;
        for (uint row = 0; row < OTC_ROWS; ++row)
            pixels[row] = denominator[row] == 0.0f ? (ushort)0 :
                intel_convert_bfloat16_as_ushort(
                    accumulator[depth][row] / denominator[row]);
        intel_sub_group_2d_block_write_16b_8r16x1c(
            (__global void *)(output + target), 256, 8, 256,
            (int2)(depth * 16, 0), (__private ushort *)&pixels);
    }
#endif
#endif
}
#endif

#if defined(OTC_SPLIT_PARTIAL)
__attribute__((intel_reqd_sub_group_size(OTC_GROUP_SIZE)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void otc_plus_split_key_partial(const __global ushort *q,
                                          const __global ushort *k,
                                          const __global ushort *v,
                                          const __global int *blocks,
                                          __global float *partial) {
    const uint lane = get_sub_group_local_id();
#if defined(OTC_A770_SG8)
    // Repeat-eight DPAS halves duplicate K/V loads for the same eight-row
    // output tile. Keep the shared XML/ABI; no work-group barriers are used.
    if (get_sub_group_id() != 0) return;
#endif
    const uint q_heads = (uint)INPUT0_DIMS[0];
    const uint out_head = get_group_id(2);
    if (q_heads == 0 || out_head >= OTC_SPLIT_KEY * q_heads)
        return;
    const uint split = out_head / q_heads;
    const uint head = out_head % q_heads;
    const uint window = get_group_id(1) / 16;
    const uint first_query = (get_group_id(1) % 16) * 8;
    const bool canonical_kv = INPUT1_DIMS[2] == INPUT0_DIMS[0];
    const uint key_windows = canonical_kv ? INPUT1_DIMS[0] : INPUT1_DIMS[1];
    const bool meta_ok = q_heads <= 12 && INPUT0_DIMS[2] == 128 &&
        INPUT0_DIMS[3] == 128 && INPUT1_DIMS[3] == 128 &&
        INPUT2_DIMS[3] == 128 && INPUT3_DIMS[0] == q_heads &&
        INPUT3_DIMS[1] == INPUT0_DIMS[1] && INPUT3_DIMS[2] == 1 &&
        INPUT3_PITCHES[3] == 1 && INPUT3_PITCHES[2] == INPUT3_DIMS[3] &&
        INPUT3_PITCHES[1] == INPUT3_DIMS[3] &&
        INPUT3_PITCHES[0] == INPUT3_DIMS[1] * INPUT3_DIMS[3] &&
        OUTPUT0_DIMS[0] == OTC_SPLIT_KEY * q_heads &&
        OUTPUT0_DIMS[1] == INPUT0_DIMS[1] && OUTPUT0_DIMS[2] == 128 &&
        OUTPUT0_DIMS[3] == 130 && INPUT0_PITCHES[3] == 1 &&
        INPUT0_PITCHES[2] == 128 && OUTPUT0_PITCHES[3] == 1 &&
        OUTPUT0_PITCHES[2] == 130;
    const size_t qbase = INPUT0_OFFSET + head * INPUT0_PITCHES[0] +
        window * INPUT0_PITCHES[1] + first_query * INPUT0_PITCHES[2];
    const size_t target = OUTPUT0_OFFSET + out_head * OUTPUT0_PITCHES[0] +
        window * OUTPUT0_PITCHES[1] + first_query * OUTPUT0_PITCHES[2];
    const size_t selection = INPUT3_OFFSET + head * INPUT3_PITCHES[0] +
        window * INPUT3_PITCHES[1];
    if (!meta_ok || (((ulong)(q + qbase) | (ulong)(k + INPUT1_OFFSET) |
          (ulong)(v + INPUT2_OFFSET) | (ulong)(partial + target)) & 63) != 0 ||
        ((ulong)(blocks + selection) & 3) != 0 || INPUT1_PITCHES[3] != 1 ||
        INPUT2_PITCHES[3] != 1 || INPUT1_PITCHES[2] != 128 ||
        INPUT2_PITCHES[2] != 128 || (canonical_kv &&
        (INPUT1_PITCHES[1] != INPUT0_DIMS[0] * 128 ||
         INPUT2_PITCHES[1] != INPUT0_DIMS[0] * 128)))
        return;
    uint valid_slots = 0;
#ifdef OTC_BINARY_PREFIX
    uint prefix_end = INPUT3_DIMS[3];
    while (valid_slots < prefix_end) {
        const uint middle = valid_slots + ((prefix_end - valid_slots) >> 1);
        if (blocks[selection + middle * INPUT3_PITCHES[3]] >= 0)
            valid_slots = middle + 1;
        else
            prefix_end = middle;
    }
#else
    while (valid_slots < INPUT3_DIMS[3] &&
           blocks[selection + valid_slots * INPUT3_PITCHES[3]] >= 0)
        ++valid_slots;
#endif
    otc_run_attention(q, k, v, blocks, partial, head, window, first_query,
                      (valid_slots * split) / OTC_SPLIT_KEY,
                      (valid_slots * (split + 1)) / OTC_SPLIT_KEY,
                      canonical_kv, key_windows, qbase, target, selection, lane);
}

#elif defined(OTC_SPLIT_MERGE)
__attribute__((intel_reqd_sub_group_size(OTC_GROUP_SIZE)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void otc_plus_split_key_merge(const __global float *partial,
                                        __global ushort *output) {
    const uint lane = get_sub_group_local_id();
    const uint heads = (uint)OUTPUT0_DIMS[0];
    const uint head = get_group_id(2);
    const uint window = get_group_id(1) / 16;
    const uint first_query = (get_group_id(1) % 16) * 8
#if defined(OTC_A770_SG8)
        + get_sub_group_id() * 4
#endif
        ;
    const bool meta_ok = heads >= 1 && heads <= 12 && OUTPUT0_DIMS[2] == 128 &&
        OUTPUT0_DIMS[3] == 128 && INPUT0_DIMS[0] == OTC_SPLIT_KEY * heads &&
        INPUT0_DIMS[1] == OUTPUT0_DIMS[1] && INPUT0_DIMS[2] == 128 &&
        INPUT0_DIMS[3] == 130 && OUTPUT0_PITCHES[3] == 1 &&
        OUTPUT0_PITCHES[2] == 128 && INPUT0_PITCHES[3] == 1 &&
        INPUT0_PITCHES[2] == 130 && head < heads;
    const size_t target = OUTPUT0_OFFSET + head * OUTPUT0_PITCHES[0] +
        window * OUTPUT0_PITCHES[1] + first_query * OUTPUT0_PITCHES[2];
    if (!meta_ok || (((ulong)(partial + INPUT0_OFFSET + window * INPUT0_PITCHES[1] +
          (first_query - get_sub_group_id() * OTC_ROWS) * INPUT0_PITCHES[2]) |
          (ulong)(output + target)) & 63) != 0)
        return;
    float global_max[OTC_ROWS];
    for (uint row = 0; row < OTC_ROWS; ++row) {
        float best = -INFINITY;
        for (uint split = 0; split < OTC_SPLIT_KEY; ++split) {
            const uint out_head = split * heads + head;
            const size_t at = INPUT0_OFFSET + out_head * INPUT0_PITCHES[0] +
                window * INPUT0_PITCHES[1] + (first_query + row) * INPUT0_PITCHES[2];
            best = fmax(best, partial[at + 128 * INPUT0_PITCHES[3]]);
        }
        global_max[row] = best;
    }
    for (uint depth = 0; depth < OTC_OUT_TILES; ++depth) {
#if !defined(OTC_A770_SG8)
        ushort8 pixels;
#endif
        for (uint row = 0; row < OTC_ROWS; ++row) {
            float denominator = 0.0f;
            float numerator = 0.0f;
            if (global_max[row] != -INFINITY)
                for (uint split = 0; split < OTC_SPLIT_KEY; ++split) {
                    const uint out_head = split * heads + head;
                    const size_t at = INPUT0_OFFSET + out_head * INPUT0_PITCHES[0] +
                        window * INPUT0_PITCHES[1] + (first_query + row) * INPUT0_PITCHES[2];
                    const float local_max = partial[at + 128 * INPUT0_PITCHES[3]];
                    const float local_den = partial[at + 129 * INPUT0_PITCHES[3]];
                    if (local_den > 0.0f) {
                        const float weight = exp(local_max - global_max[row]);
                        denominator += weight * local_den;
                        numerator += weight * partial[at +
                            (depth * OTC_GROUP_SIZE + lane) * INPUT0_PITCHES[3]];
                    }
                }
            const ushort pixel = denominator > 0.0f
                ? intel_convert_bfloat16_as_ushort(numerator / denominator)
                : (ushort)0;
#if defined(OTC_A770_SG8)
            output[target + row * OUTPUT0_PITCHES[2] +
                   (depth * OTC_GROUP_SIZE + lane) * OUTPUT0_PITCHES[3]] = pixel;
#else
            pixels[row] = pixel;
#endif
        }
#if !defined(OTC_A770_SG8)
        intel_sub_group_2d_block_write_16b_8r16x1c(
            (__global void *)(output + target), 256, 8, 256,
            (int2)(depth * 16, 0), (__private ushort *)&pixels);
#endif
    }
}

#else
// Direct sparse entry.  OTC_CROSS_PV_SPLIT keeps the old precise-cross name
// and only changes the residual BF16 numerator accumulation in the shared
// loop.  No arithmetic-ordering claim is made for the SG8 fallback: its
// eight-lane reduction and packed probability shuffles are intentionally a
// different DPAS decomposition from the qualified SIMD16 path.
#if defined(OTC_CROSS_PV_SPLIT)
#define OTC_DIRECT_NAME otc_cross_precise_xmx
#else
#define OTC_DIRECT_NAME otc_plus_sparse_xmx
#endif
__attribute__((intel_reqd_sub_group_size(OTC_GROUP_SIZE)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void OTC_DIRECT_NAME(const __global ushort *q,
                              const __global ushort *k,
                              const __global ushort *v,
                              const __global int *blocks,
                              __global ushort *output) {
    const uint lane = get_sub_group_local_id();
    const uint head = get_group_id(2);
    const uint window = get_group_id(1) / 16;
    const uint first_query = (get_group_id(1) % 16) * 8
#if defined(OTC_A770_SG8)
        + get_sub_group_id() * 4
#endif
        ;
    const bool canonical_kv = INPUT1_DIMS[2] == INPUT0_DIMS[0];
    const uint key_windows = canonical_kv ? INPUT1_DIMS[0] : INPUT1_DIMS[1];
    const size_t qbase = INPUT0_OFFSET + head * INPUT0_PITCHES[0] +
        window * INPUT0_PITCHES[1] + first_query * INPUT0_PITCHES[2];
    const size_t target = OUTPUT0_OFFSET + head * OUTPUT0_PITCHES[0] +
        window * OUTPUT0_PITCHES[1] + first_query * OUTPUT0_PITCHES[2];
    const size_t selection = INPUT3_OFFSET + head * INPUT3_PITCHES[0] +
        window * INPUT3_PITCHES[1];
    const bool meta_ok = INPUT0_DIMS[2] == 128 && INPUT0_DIMS[3] == 128 &&
        INPUT1_DIMS[3] == 128 && INPUT2_DIMS[3] == 128 &&
        INPUT0_PITCHES[2] == 128 && INPUT1_PITCHES[2] == 128 &&
        INPUT2_PITCHES[2] == 128 && OUTPUT0_PITCHES[2] == 128 &&
        INPUT0_PITCHES[3] == 1 && INPUT1_PITCHES[3] == 1 &&
        INPUT2_PITCHES[3] == 1 && OUTPUT0_PITCHES[3] == 1 &&
        (!canonical_kv || (INPUT1_PITCHES[1] == INPUT0_DIMS[0] * 128 &&
                           INPUT2_PITCHES[1] == INPUT0_DIMS[0] * 128));
    if (!meta_ok || (((ulong)(q + qbase) | (ulong)(k + INPUT1_OFFSET) |
          (ulong)(v + INPUT2_OFFSET) | (ulong)(output + target)) & 63) != 0) {
#if defined(OTC_A770_SG8)
        for (uint row = 0; row < OTC_ROWS; ++row)
            for (uint depth = 0; depth < OTC_OUT_TILES; ++depth)
                output[target + row * OUTPUT0_PITCHES[2] +
                       (depth * OTC_GROUP_SIZE + lane) * OUTPUT0_PITCHES[3]] = 0;
#else
        for (uint row = 0; row < OTC_ROWS; ++row)
            for (uint depth = 0; depth < OTC_OUT_TILES; ++depth)
                output[target + row * OUTPUT0_PITCHES[2] +
                       (depth * 16 + lane) * OUTPUT0_PITCHES[3]] = 0;
#endif
        return;
    }
    otc_run_attention(q, k, v, blocks, output, head, window, first_query, 0,
                      INPUT3_DIMS[3], canonical_kv, key_windows, qbase, target,
                      selection, lane);
}
#endif
