// Lossless BF16 bridge draft for the already-qualified raw-BF16 XMX layer.
//
// The bridge CustomLayers receive numeric F32 values that came directly from
// an OV BF16 Convert.  Every accepted F32 word therefore has zero low 16
// bits; the bridge writes/reads those upper 16 bits as ushort storage exposed
// through an F16 port.  No half arithmetic, conversion, or transpose occurs
// in this file.  Invalid layout/value metadata writes a non-finite sentinel
// instead of leaving stale output for a later consumer to mistake as valid.
#if defined(OTC_CROSS_Q_PACK)

#define Q_ELEMS 3145728u
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void otc_cross_q_pack(const __global float *src,
                               __global ushort *dst) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= Q_ELEMS)
        return;
    const uint row = gid / 1536u;
    const uint channel = gid % 1536u;
    const uint head = channel / 128u;
    const uint depth = channel % 128u;
    const uint window = row / 128u;
    const uint token = row % 128u;
    const size_t so = INPUT0_OFFSET + (size_t)row * INPUT0_PITCHES[2]
        + (size_t)channel * INPUT0_PITCHES[3];
    const size_t doff = OUTPUT0_OFFSET + (size_t)head * OUTPUT0_PITCHES[0]
        + (size_t)window * OUTPUT0_PITCHES[1]
        + (size_t)token * OUTPUT0_PITCHES[2]
        + (size_t)depth * OUTPUT0_PITCHES[3];
    const bool shape_ok = INPUT0_DIMS[0] == 1 && INPUT0_DIMS[1] == 1
        && INPUT0_DIMS[2] == 2048 && INPUT0_DIMS[3] == 1536 && OUTPUT0_DIMS[0] == 12
        && OUTPUT0_DIMS[1] == 16 && OUTPUT0_DIMS[2] == 128
        && OUTPUT0_DIMS[3] == 128 && INPUT0_PITCHES[3] == 1
        && OUTPUT0_PITCHES[3] == 1 && OUTPUT0_PITCHES[2] == 128
        && (((ulong)(src + INPUT0_OFFSET) | (ulong)(dst + OUTPUT0_OFFSET)) & 63) == 0;
    const uint bits = shape_ok ? as_uint(src[so]) : 0x7FC10000u;
    const bool exact_finite_bf16 = (bits & 0xFFFFu) == 0
        && ((bits >> 16) & 0x7F80u) != 0x7F80u;
    dst[doff] = (!shape_ok || !exact_finite_bf16)
        ? (ushort)0x7FC1 : (ushort)(bits >> 16);
}

#elif defined(OTC_CROSS_KV_PACK)

#define KV_ELEMS 786432u
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void otc_cross_kv_pack(const __global float *src,
                                __global ushort *dst) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= KV_ELEMS)
        return;
    const uint row = gid / 1536u;
    const uint channel = gid % 1536u;
    const uint block = row / 128u;
    const uint token = row % 128u;
    const uint head = channel / 128u;
    const uint depth = channel % 128u;
    const size_t so = INPUT0_OFFSET + (size_t)row * INPUT0_PITCHES[2]
        + (size_t)channel * INPUT0_PITCHES[3];
    const size_t doff = OUTPUT0_OFFSET + (size_t)block * OUTPUT0_PITCHES[0]
        + (size_t)token * OUTPUT0_PITCHES[1]
        + (size_t)head * OUTPUT0_PITCHES[2]
        + (size_t)depth * OUTPUT0_PITCHES[3];
    const bool shape_ok = INPUT0_DIMS[0] == 1 && INPUT0_DIMS[1] == 1
        && INPUT0_DIMS[2] == 512 && INPUT0_DIMS[3] == 1536 && OUTPUT0_DIMS[0] == 4
        && OUTPUT0_DIMS[1] == 128 && OUTPUT0_DIMS[2] == 12
        && OUTPUT0_DIMS[3] == 128 && INPUT0_PITCHES[3] == 1
        && OUTPUT0_PITCHES[3] == 1 && OUTPUT0_PITCHES[2] == 128
        && (((ulong)(src + INPUT0_OFFSET) | (ulong)(dst + OUTPUT0_OFFSET)) & 63) == 0;
    const uint bits = shape_ok ? as_uint(src[so]) : 0x7FC10000u;
    const bool exact_finite_bf16 = (bits & 0xFFFFu) == 0
        && ((bits >> 16) & 0x7F80u) != 0x7F80u;
    dst[doff] = (!shape_ok || !exact_finite_bf16)
        ? (ushort)0x7FC1 : (ushort)(bits >> 16);
}

#elif defined(OTC_CROSS_CONTEXT_UNPACK)

#define CONTEXT_ELEMS 3145728u
__attribute__((reqd_work_group_size(16, 1, 1)))
__kernel void otc_cross_context_unpack(const __global ushort *src,
                                       __global float *dst) {
    const uint gid = (uint)get_global_id(0);
    if (gid >= CONTEXT_ELEMS)
        return;
    const uint row = gid / 1536u;
    const uint channel = gid % 1536u;
    const uint head = channel / 128u;
    const uint depth = channel % 128u;
    const uint window = row / 128u;
    const uint token = row % 128u;
    const size_t so = INPUT0_OFFSET + (size_t)head * INPUT0_PITCHES[0]
        + (size_t)window * INPUT0_PITCHES[1]
        + (size_t)token * INPUT0_PITCHES[2]
        + (size_t)depth * INPUT0_PITCHES[3];
    const size_t doff = OUTPUT0_OFFSET + (size_t)row * OUTPUT0_PITCHES[2]
        + (size_t)channel * OUTPUT0_PITCHES[3];
    const bool shape_ok = INPUT0_DIMS[0] == 12 && INPUT0_DIMS[1] == 16
        && INPUT0_DIMS[2] == 128 && INPUT0_DIMS[3] == 128
        && OUTPUT0_DIMS[0] == 1 && OUTPUT0_DIMS[1] == 1
        && OUTPUT0_DIMS[2] == 2048 && OUTPUT0_DIMS[3] == 1536 && INPUT0_PITCHES[3] == 1
        && INPUT0_PITCHES[2] == 128 && OUTPUT0_PITCHES[3] == 1
        && (((ulong)(src + INPUT0_OFFSET) | (ulong)(dst + OUTPUT0_OFFSET)) & 63) == 0;
    const uint raw = shape_ok ? (uint)src[so] : 0x7FC1u;
    const bool finite = ((raw & 0x7F80u) != 0x7F80u);
    dst[doff] = (!shape_ok || !finite)
        ? as_float((uint)0x7FC10000) : as_float(raw << 16);
}

#else
#error "Select exactly one BF16 bridge kernel"
#endif

