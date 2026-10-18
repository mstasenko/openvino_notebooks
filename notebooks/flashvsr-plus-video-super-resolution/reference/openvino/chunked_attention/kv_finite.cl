// Raw F16 finite-bit scan for one complete canonical K/V cache window.
// Each 256-lane workgroup owns one B/window and emits one exact F16 flag.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void otc_complete_kv_finite(const __global ushort *k,
                                     const __global ushort *v,
                                     __global ushort *flags) {
    const ulong lane = (ulong)get_local_id(0);
    const ulong window = (ulong)get_group_id(2);
    const ulong max_u64 = (ulong)(-1);

    const bool input_meta_ok =
        INPUT0_DIMS[0] == INPUT1_DIMS[0] &&
        INPUT0_DIMS[1] == 128 && INPUT0_DIMS[2] == 12 && INPUT0_DIMS[3] == 128 &&
        INPUT1_DIMS[1] == 128 && INPUT1_DIMS[2] == 12 && INPUT1_DIMS[3] == 128 &&
        INPUT0_PITCHES[3] == 1 && INPUT0_PITCHES[2] == 128 &&
        INPUT0_PITCHES[1] == 1536 && INPUT0_PITCHES[0] == 196608 &&
        INPUT1_PITCHES[3] == 1 && INPUT1_PITCHES[2] == 128 &&
        INPUT1_PITCHES[1] == 1536 && INPUT1_PITCHES[0] == 196608;
    const bool output_meta_ok =
        OUTPUT0_DIMS[0] == INPUT0_DIMS[0] && OUTPUT0_DIMS[1] == 1 &&
        OUTPUT0_DIMS[2] == 1 && OUTPUT0_DIMS[3] == 1 &&
        OUTPUT0_PITCHES[3] == 1 && OUTPUT0_PITCHES[2] == 1 &&
        OUTPUT0_PITCHES[1] == 1 && OUTPUT0_PITCHES[0] == 1;
    const bool window_ok = window < (ulong)INPUT0_DIMS[0];

    const ulong elements = (ulong)128 * (ulong)12 * (ulong)128;
    const ulong k_pitch = (ulong)INPUT0_PITCHES[0];
    const ulong v_pitch = (ulong)INPUT1_PITCHES[0];
    const bool k_row_ok = input_meta_ok && window_ok &&
                          (window == 0 || k_pitch <= max_u64 / window);
    const bool v_row_ok = input_meta_ok && window_ok &&
                          (window == 0 || v_pitch <= max_u64 / window);
    ulong k_row = 0;
    ulong v_row = 0;
    if (k_row_ok)
        k_row = window * k_pitch;
    if (v_row_ok)
        v_row = window * v_pitch;

    ulong k_base = 0;
    ulong v_base = 0;
    bool k_base_ok = k_row_ok &&
                     k_row <= max_u64 - (ulong)INPUT0_OFFSET;
    bool v_base_ok = v_row_ok &&
                     v_row <= max_u64 - (ulong)INPUT1_OFFSET;
    if (k_base_ok) {
        k_base = (ulong)INPUT0_OFFSET + k_row;
        k_base_ok = elements <= max_u64 - k_base;
    }
    if (v_base_ok) {
        v_base = (ulong)INPUT1_OFFSET + v_row;
        v_base_ok = elements <= max_u64 - v_base;
    }
    const bool offsets_ok = k_base_ok && v_base_ok;

    // Do not form or write an output address until its metadata and the
    // window-times-pitch multiplication have both been proven safe.
    const ulong output_pitch = (ulong)OUTPUT0_PITCHES[0];
    const bool output_row_ok = output_meta_ok && window_ok &&
                               (window == 0 || output_pitch <= max_u64 / window);
    ulong out = 0;
    bool output_address_ok = output_row_ok;
    if (output_address_ok) {
        const ulong out_row = window * output_pitch;
        if (out_row <= max_u64 - (ulong)OUTPUT0_OFFSET)
            out = (ulong)OUTPUT0_OFFSET + out_row;
        else
            output_address_ok = false;
    }

    // The host pre-zeroes flags, so malformed metadata fails closed without
    // any sentinel write through an address that was not proven valid.
    if (!input_meta_ok || !output_meta_ok || !window_ok || !offsets_ok ||
        !output_address_ok)
        return;

    uint bad = 0;
    for (ulong index = lane; index < elements; index += (ulong)256) {
        const ushort k_bits = k[k_base + index];
        const ushort v_bits = v[v_base + index];
        bad |= ((k_bits & (ushort)0x7c00) == (ushort)0x7c00);
        bad |= ((v_bits & (ushort)0x7c00) == (ushort)0x7c00);
    }

    __local uint reduced[256];
    reduced[lane] = bad;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (lane < (ulong)stride)
            reduced[lane] |= reduced[lane + (ulong)stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0)
        // Finite is exactly 1.0 F16; non-finite is exactly 0.0 F16.
        flags[out] = reduced[0] ? (ushort)0x0000 : (ushort)0x3c00;
}
