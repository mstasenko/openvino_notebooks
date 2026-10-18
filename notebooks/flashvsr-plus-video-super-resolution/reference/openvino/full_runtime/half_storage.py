"""Bounded validity checks without slow elementwise FP16 conversion."""
import numpy as np


def finite_half(value):
    """FP16 exponent 31 is exactly the Inf/NaN set, including either sign."""
    if value.dtype!=np.float16:raise ValueError('Expected original half storage')
    bits=value.reshape(-1).view(np.uint16)
    # Masking the sign makes max >= 0x7c00 exactly the Inf/NaN set,
    # without allocating an additional chunk-sized boolean comparison.
    return all(np.max(bits[start:start+(1<<20)]&np.uint16(0x7fff))<np.uint16(0x7c00)
               for start in range(0,bits.size,1<<20))


def all_finite(value):
    return finite_half(value)if value.dtype==np.float16 else bool(np.isfinite(value).all())
