"""Narrow official move-only BlockSwap: no RoPE retry/protection patches."""
import weakref


def configure_prefix(model, count, device, offload, debug, configure=None, wrap=None):
    if type(count) is not int or count not in (0, 16) or len(model.blocks) != 32:
        raise ValueError("Only 32-block N0/N16 controls are qualified here")
    weakref.ref(debug)  # Upstream wrappers hold weak refs; caller keeps Debug alive.
    if any(hasattr(block, "_original_forward") for block in model.blocks):
        raise ValueError("Refusing already-wrapped block")
    if any(hasattr(model, name) for name in ("blocks_to_swap", "main_device", "offload_device")):
        raise ValueError("Refusing previously configured model")
    if count == 0:
        return {"selected_blocks": [], "last_selected_index": None}
    if configure is None or wrap is None:
        if configure is not None or wrap is not None:
            raise ValueError("Placement/wrapper hooks must be provided together")
        from src.optimization.blockswap import _configure_blocks, _wrap_block_forward
        configure, wrap = _configure_blocks, _wrap_block_forward
    model.blocks_to_swap = count - 1
    model.main_device = device
    model.offload_device = offload
    configure(model, device, offload, debug)
    for index in range(count):
        wrap(model.blocks[index], index, model, debug)
    return {"selected_blocks": list(range(count)), "last_selected_index": count - 1}
