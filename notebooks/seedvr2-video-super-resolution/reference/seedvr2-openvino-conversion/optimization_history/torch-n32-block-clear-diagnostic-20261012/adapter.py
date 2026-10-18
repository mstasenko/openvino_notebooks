"""N32-only additive hook for a post-offload public XPU cache clear.

The stock path delegates unchanged to the authenticated upstream placement and
wrapper functions.  The diagnostic path wraps that existing wrapper only after
it returns: the upstream wrapper has already moved the block to CPU, while its
returned activation is held by ``output`` here.  A caller-provided synchronize
and boundary callback then perform the measured forced minimal clear.
"""

import types
import weakref


def _hook_after_offload(block, model, synchronize, boundary):
    """Add the diagnostic seam around one already-installed official wrapper."""
    upstream_function = block.forward.__func__
    model_ref = weakref.ref(model)

    def hooked_forward(self, *args, **kwargs):
        # Keep the upstream function object strong, but do not retain a bound
        # method cycle after replacing block.forward.
        output = upstream_function(self, *args, **kwargs)
        owner = model_ref()
        if owner is not None and self._block_idx <= owner.blocks_to_swap:
            # The official wrapper has completed its blocking CPU offload.  Keep
            # output live through both callbacks; neither callback may read it.
            if synchronize is not None:
                synchronize()
            boundary(self._block_idx)
        return output

    block.forward = types.MethodType(hooked_forward, block)


def configure_prefix(
    model,
    count,
    device,
    offload,
    debug,
    variant="stock",
    synchronize=None,
    boundary=None,
    configure=None,
    wrap=None,
):
    """Configure exactly 32 blocks, optionally adding the isolated seam hook.

    ``configure`` and ``wrap`` are injectable only for source tests.  In a real
    run they resolve to the exact upstream primitives from ``blockswap.py``.
    """
    if variant not in ("stock", "block-clear"):
        raise ValueError("Only stock and block-clear diagnostics are admitted")
    if type(count) is not int or count != 32 or len(model.blocks) != 32:
        raise ValueError("This diagnostic admits N32 only")
    weakref.ref(debug)  # Match the upstream wrapper lifetime contract.
    if any(hasattr(block, "_original_forward") for block in model.blocks):
        raise ValueError("Refusing already-wrapped block")
    if any(hasattr(model, name) for name in ("blocks_to_swap", "main_device", "offload_device")):
        raise ValueError("Refusing previously configured model")
    if boundary is not None and not callable(boundary):
        raise ValueError("boundary must be callable when supplied")
    if variant == "block-clear" and (not callable(synchronize) or boundary is None):
        raise ValueError("block-clear requires synchronize and boundary callbacks")
    if variant == "stock" and synchronize is not None:
        raise ValueError("stock observation must not add synchronization")
    if configure is None or wrap is None:
        if configure is not None or wrap is not None:
            raise ValueError("Placement/wrapper hooks must be provided together")
        from src.optimization.blockswap import _configure_blocks, _wrap_block_forward
        configure, wrap = _configure_blocks, _wrap_block_forward

    model.blocks_to_swap = 31
    model.main_device = device
    model.offload_device = offload
    placement = configure(model, device, offload, debug)
    for index in range(32):
        wrap(model.blocks[index], index, model, debug)
        if boundary is not None:
            _hook_after_offload(model.blocks[index], model, synchronize, boundary)
    return {
        "variant": variant,
        "selected_blocks": list(range(32)),
        "last_selected_index": 31,
        "upstream_placement": placement,
        "forced_clear_per_forward": 32 if variant == "block-clear" else 0,
    }
