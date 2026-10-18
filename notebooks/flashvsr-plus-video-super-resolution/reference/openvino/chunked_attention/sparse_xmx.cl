// Direct selected-block attention: no union padding, gathered K/V, score
// tensor or local-memory barriers. One SIMD16 group owns eight query rows.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void otc_sparse_xmx(const __global half *q,
                             const __global half *k,
                             const __global half *v,
                             const __global int *blocks,
                             __global half *output) {
    const uint lane = get_sub_group_local_id();
    const uint head = get_group_id(2);
    const uint window = get_group_id(1) / 16;
    const uint first_query = (get_group_id(1) % 16) * 8;
    // Both ABIs keep depth contiguous. Canonical W,T,H,D can be read with
    // strided 2D loads, avoiding a full CPU transpose before every upload.
    const bool canonical_kv = INPUT1_DIMS[2] == INPUT0_DIMS[0];
    const uint key_windows = canonical_kv ? INPUT1_DIMS[0] : INPUT1_DIMS[1];
    const uint kpitch = 2 * (canonical_kv ? INPUT1_PITCHES[1] : INPUT1_PITCHES[2]);
    const uint vpitch = 2 * (canonical_kv ? INPUT2_PITCHES[1] : INPUT2_PITCHES[2]);
    const size_t qbase = INPUT0_OFFSET + head * INPUT0_PITCHES[0]
        + window * INPUT0_PITCHES[1] + first_query * INPUT0_PITCHES[2];
    const size_t target = OUTPUT0_OFFSET + head * OUTPUT0_PITCHES[0]
        + window * OUTPUT0_PITCHES[1] + first_query * OUTPUT0_PITCHES[2];
    // 2D IO needs full SIMD16 groups and aligned contiguous BFYX rows.
    // Fail closed without issuing an undefined block operation if a backend
    // ever changes this admitted physical layout.
    if ((((ulong)(q + qbase) | (ulong)(k + INPUT1_OFFSET)
          | (ulong)(v + INPUT2_OFFSET) | (ulong)(output + target)) & 63) != 0
        || INPUT0_PITCHES[2] != 128 || INPUT1_PITCHES[2] != 128 || INPUT2_PITCHES[2] != 128
        || (canonical_kv && (INPUT1_PITCHES[1] != INPUT0_DIMS[0] * 128
                            || INPUT2_PITCHES[1] != INPUT0_DIMS[0] * 128))
        || OUTPUT0_PITCHES[2] != 128 || INPUT0_PITCHES[3] != 1 || INPUT1_PITCHES[3] != 1
        || INPUT2_PITCHES[3] != 1 || OUTPUT0_PITCHES[3] != 1) {
        for (uint row = 0; row < 8; ++row)
            for (uint depth = 0; depth < 8; ++depth)
                output[target + row * OUTPUT0_PITCHES[2] + (depth * 16 + lane) * OUTPUT0_PITCHES[3]] = (half)0;
        return;
    }
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
        if (block < 0) break; // Host admits only ordered IDs then -1 padding.
        if (block >= key_windows) continue; // Never read out of bounds.
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
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)(k + kbase),
                    256, 128, kpitch, (int2)(depth * 8, start), (__private uint *)&key0);
                intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)(k + kbase),
                    256, 128, kpitch, (int2)(depth * 8, start + 16), (__private uint *)&key1);
                logits0 = intel_sub_group_f16_f16_matrix_mad_k16(query[depth], key0, logits0);
                logits1 = intel_sub_group_f16_f16_matrix_mad_k16(query[depth], key1, logits1);
            }
            logits0 *= 0.08838834764831844f; // 1/sqrt(128), same global scale.
            logits1 *= 0.08838834764831844f;
            short8 probability0, probability1;
            float8 correction;
            #pragma unroll
            for (uint row = 0; row < 8; ++row) {
                const float updated = fmax(maximum[row], sub_group_reduce_max(fmax(logits0[row], logits1[row])));
                correction[row] = exp(maximum[row] - updated);
                const float p0 = exp(logits0[row] - updated), p1 = exp(logits1[row] - updated);
                denominator[row] = denominator[row] * correction[row] + sub_group_reduce_add(p0 + p1);
                maximum[row] = updated;
                probability0[row] = as_short(convert_half_rte(p0));
                probability1[row] = as_short(convert_half_rte(p1));
            }
            #pragma unroll
            for (uint depth = 0; depth < 8; ++depth) {
                // Materialize before the reads: the FP64 regression caught
                // first16-channel corruption with the in-place array expression.
                float8 updated = accumulator[depth] * correction;
                int8 value0, value1;
                // DPAS right operands pack adjacent key rows into dwords;
                // the ordinary 16-bit block load has a different layout.
                intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void *)(v + vbase),
                    256, 128, vpitch, (int2)(depth * 16, start), (__private uint *)&value0);
                intel_sub_group_2d_block_read_transform_16b_16r16x1c((__global void *)(v + vbase),
                    256, 128, vpitch, (int2)(depth * 16, start + 16), (__private uint *)&value1);
                // Rescale all past blocks before adding this tile. Summing
                // independently normalized blocks would be mathematically wrong.
                updated = intel_sub_group_f16_f16_matrix_mad_k16(probability0, value0, updated);
                accumulator[depth] = intel_sub_group_f16_f16_matrix_mad_k16(
                    probability1, value1, updated);
            }
        }
    }
    #pragma unroll
    for (uint depth = 0; depth < 8; ++depth) {
        ushort8 pixels;
        #pragma unroll
        for (uint row = 0; row < 8; ++row)
            pixels[row] = as_ushort(denominator[row] == 0
                ? (half)0 : convert_half_rte(accumulator[depth][row] / denominator[row]));
        intel_sub_group_2d_block_write_16b_8r16x1c((__global void *)(output + target),
            256, 8, 256, (int2)(depth * 16, 0), (__private ushort *)&pixels);
    }
}
