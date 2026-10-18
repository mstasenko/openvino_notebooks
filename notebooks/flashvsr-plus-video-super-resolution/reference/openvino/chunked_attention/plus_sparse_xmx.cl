// Direct selected-block BF16 attention.  The layout and work partition are
// the existing sparse_xmx ABI: one SIMD16 group owns eight query rows.
// BF16 has no OpenCL C scalar type in this extension; all storage is raw
// 16-bit bits (ushort/short) and matrix operands are interpreted as BF16.
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

#if defined(OTC_CROSS_PV_SPLIT)
#define otc_plus_sparse_xmx otc_cross_precise_xmx
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void otc_plus_sparse_xmx(const __global ushort *q,
                                  const __global ushort *k,
                                  const __global ushort *v,
                                  const __global int *blocks,
                                  __global ushort *output) {
    const uint lane = get_sub_group_local_id();
    const uint head = get_group_id(2);
    const uint window = get_group_id(1) / 16;
    const uint first_query = (get_group_id(1) % 16) * 8;
    // Both ABIs keep depth contiguous.  A ushort has the same two-byte
    // pitch as the original 16-bit port, so the existing 2D block geometry
    // applies unchanged.
    const bool canonical_kv = INPUT1_DIMS[2] == INPUT0_DIMS[0];
    const uint key_windows = canonical_kv ? INPUT1_DIMS[0] : INPUT1_DIMS[1];
    const uint kpitch = 2 * (canonical_kv ? INPUT1_PITCHES[1] : INPUT1_PITCHES[2]);
    const uint vpitch = 2 * (canonical_kv ? INPUT2_PITCHES[1] : INPUT2_PITCHES[2]);
    const size_t qbase = INPUT0_OFFSET + head * INPUT0_PITCHES[0]
        + window * INPUT0_PITCHES[1] + first_query * INPUT0_PITCHES[2];
    const size_t target = OUTPUT0_OFFSET + head * OUTPUT0_PITCHES[0]
        + window * OUTPUT0_PITCHES[1] + first_query * OUTPUT0_PITCHES[2];
    // 2D IO requires full SIMD16 groups and aligned contiguous rows.  The
    // host admission path produces this physical layout; fail closed if a
    // backend changes it rather than issuing an undefined block operation.
    if ((((ulong)(q + qbase) | (ulong)(k + INPUT1_OFFSET)
          | (ulong)(v + INPUT2_OFFSET) | (ulong)(output + target)) & 63) != 0
        || INPUT0_PITCHES[2] != 128 || INPUT1_PITCHES[2] != 128 || INPUT2_PITCHES[2] != 128
        || (canonical_kv && (INPUT1_PITCHES[1] != INPUT0_DIMS[0] * 128
                            || INPUT2_PITCHES[1] != INPUT0_DIMS[0] * 128))
        || OUTPUT0_PITCHES[2] != 128 || INPUT0_PITCHES[3] != 1 || INPUT1_PITCHES[3] != 1
        || INPUT2_PITCHES[3] != 1 || OUTPUT0_PITCHES[3] != 1) {
        for (uint row = 0; row < 8; ++row)
            for (uint depth = 0; depth < 8; ++depth)
                output[target + row * OUTPUT0_PITCHES[2]
                       + (depth * 16 + lane) * OUTPUT0_PITCHES[3]] = (ushort)0;
        return;
    }

    // The BF16 matrix API takes signed short/int vectors containing raw BF16
    // bits.  Accumulation and online normalization remain FP32.
    short8 query[8];
    float8 accumulator[8];
    #pragma unroll
    for (uint tile = 0; tile < 8; ++tile) {
        short8 fragment;
        intel_sub_group_2d_block_read_16b_8r16x1c((__global void *)(q + qbase),
            256, 8, 256, (int2)(tile * 16, 0), (__private ushort *)&fragment);
        query[tile] = fragment;
        accumulator[tile] = (float8)(0);
    }
    float8 maximum = (float8)(-INFINITY);
    float8 denominator = (float8)(0);
    const size_t selection = INPUT3_OFFSET + head * INPUT3_PITCHES[0]
        + window * INPUT3_PITCHES[1];
    for (uint slot = 0; slot < INPUT3_DIMS[3]; ++slot) {
        const int block = blocks[selection + slot * INPUT3_PITCHES[3]];
        if (block < 0) break;  // Ordered IDs then -1 padding is host-admitted.
        if (block >= key_windows) continue;
        const size_t kbase = INPUT1_OFFSET + (canonical_kv
            ? (size_t)block * INPUT1_PITCHES[0] + head * INPUT1_PITCHES[2]
            : head * INPUT1_PITCHES[0] + (size_t)block * INPUT1_PITCHES[1]);
        const size_t vbase = INPUT2_OFFSET + (canonical_kv
            ? (size_t)block * INPUT2_PITCHES[0] + head * INPUT2_PITCHES[2]
            : head * INPUT2_PITCHES[0] + (size_t)block * INPUT2_PITCHES[1]);
        for (uint start = 0; start < 128; start += 32) {
            float8 logits0 = (float8)(0), logits1 = (float8)(0);
            #pragma unroll
            for (uint depth = 0; depth < 8; ++depth) {
                int8 key0, key1;
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                    (__global void *)(k + kbase), 256, 128, kpitch,
                    (int2)(depth * 8, start), (__private uint *)&key0);
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
                    (__global void *)(k + kbase), 256, 128, kpitch,
                    (int2)(depth * 8, start + 16), (__private uint *)&key1);
                logits0 = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    query[depth], key0, logits0);
                logits1 = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    query[depth], key1, logits1);
            }
            logits0 *= 0.08838834764831844f;  // 1/sqrt(128).
            logits1 *= 0.08838834764831844f;
#if defined(OTC_CROSS_PV_SPLIT)
            short8 probability0, probability1, residualprob0, residualprob1;
#else
            short8 probability0, probability1;
#endif
            float8 correction;
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                const float updated = fmax(
                    maximum[row],
                    sub_group_reduce_max(fmax(logits0[row], logits1[row])));
                correction[row] = exp(maximum[row] - updated);
                const float p0 = exp(logits0[row] - updated);
                const float p1 = exp(logits1[row] - updated);
                denominator[row] = denominator[row] * correction[row]
                    + sub_group_reduce_add(p0 + p1);
                maximum[row] = updated;
                // DPAS consumes BF16 probabilities, with RNE conversion from
                // the FP32 online-softmax values and no hidden FP16 fallback.
#if defined(OTC_CROSS_PV_SPLIT)
                const ushort hi0 = intel_convert_bfloat16_as_ushort(p0);
                const ushort hi1 = intel_convert_bfloat16_as_ushort(p1);
                probability0[row] = as_short(hi0);
                probability1[row] = as_short(hi1);
                // A second BF16 term preserves the F32 online-softmax
                // numerator more closely without changing its denominator.
                residualprob0[row] = as_short(intel_convert_bfloat16_as_ushort(
                    p0 - as_float((uint)as_ushort(hi0) << 16)));
                residualprob1[row] = as_short(intel_convert_bfloat16_as_ushort(
                    p1 - as_float((uint)as_ushort(hi1) << 16)));
#else
                probability0[row] = as_short(
                    intel_convert_bfloat16_as_ushort(p0));
                probability1[row] = as_short(
                    intel_convert_bfloat16_as_ushort(p1));
#endif
            }
            #pragma unroll
            for (uint depth = 0; depth < 8; ++depth) {
                // Materialize before reads; this preserves the old update
                // ordering while rescaling all prior blocks exactly once.
                float8 updated = accumulator[depth] * correction;
                int8 value0, value1;
                intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                    (__global void *)(v + vbase), 256, 128, vpitch,
                    (int2)(depth * 16, start), (__private uint *)&value0);
                intel_sub_group_2d_block_read_transform_16b_16r16x1c(
                    (__global void *)(v + vbase), 256, 128, vpitch,
                    (int2)(depth * 16, start + 16), (__private uint *)&value1);
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    probability0, value0, updated);
#if defined(OTC_CROSS_PV_SPLIT)
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    probability1, value1, updated);
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    residualprob0, value0, updated);
                updated = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    residualprob1, value1, updated);
#else
                accumulator[depth] = intel_sub_group_bf16_bf16_matrix_mad_k16(
                    probability1, value1, updated);
#endif
#if defined(OTC_CROSS_PV_SPLIT)
                accumulator[depth] = updated;
#endif
            }
        }
    }
    #pragma unroll
    for (uint depth = 0; depth < 8; ++depth) {
        ushort8 pixels;
        #pragma unroll
        for (uint row = 0; row < 8; ++row)
            pixels[row] = denominator[row] == 0.0f
                ? (ushort)0
                : intel_convert_bfloat16_as_ushort(
                    accumulator[depth][row] / denominator[row]);
        intel_sub_group_2d_block_write_16b_8r16x1c(
            (__global void *)(output + target), 256, 8, 256,
            (int2)(depth * 16, 0), (__private ushort *)&pixels);
    }
}
