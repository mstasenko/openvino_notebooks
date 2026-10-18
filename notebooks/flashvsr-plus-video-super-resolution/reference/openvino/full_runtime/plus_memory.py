"""Named-buffer inventory for bounded FlashVSR+ admission.

This is an explicit lower-bound inventory, not an OpenVINO allocator or peak
memory proof. The caller must provide allowances for graph workspaces, packed
upload copies, and source/diagnostic staging not named here, then select the
phase grouping proved by the controller lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from frame_geometry import FrameGeometry, frame_pair_tiles, history_memory_tiles
from plus_modes import resolve_plus_mode


HEADS = 12
HEAD_DIM = 128
BLOCK_TOKENS = 128
CHANNELS = 1536
FFN_HIDDEN = 8960
MAX_WORKSPACE_ROWS = 4096
GLOBAL_LAYERS, LOCAL_LAYERS = 5, 25
GLOBAL_RETENTION, LOCAL_RETENTION = 3, 1
CURRENT_TEMPORAL_WINDOWS = 2       # four latent frames / two source windows
ATTENTION_QUERY_WINDOWS = 16       # fixed resident XMX query chunk
PROJECTION_INPUT_CHANNELS = 768
PROJECTION_CACHE_CHANNELS = 2048
DECODER_PARAMETER_BYTES_F32 = 181_353_484  # authenticated 66-tensor TCDecoder
DECODER_SOURCE_FRAMES = 33                 # retained common input in run_plus_resident
DECODER_CHUNKS = 2
DECODER_CHUNK_FRAMES = 16
DECODER_OUTPUT_FRAMES = 29                 # 13 startup + 16 steady before final trim
DECODER_FINAL_FRAMES = 17
DECODER_STREAM_MIN_FRAMES = 17
DECODER_STREAM_MAX_FRAMES = 49
DECODER_STREAM_MAX_WORKING_SIDE = 1280
DECODER_PRIVATE_RESIDUAL_GROUPS = ((3, 512, 8), (3, 256, 4), (6, 128, 2))
# The phased builder emits one graph at a time in the UHD lifecycle.  These
# are source parameter counts in each graph after direct F32 -> BF16 storage:
# front owns blocks 0..13, each history child owns half of TGrow 15 plus
# blocks 16..19, and each frame child owns half of TGrow 21 plus blocks 22..26.
DECODER_PHASE_WEIGHT_BYTES_BF16 = {
    "front": 85_621_248,
    "history": 4_262_144,
    "frame": 629_510,
}
# Source parameter bytes for the staged history controller.  ``grow`` owns
# block 15's one output-channel half plus block 16; each memory stage owns one
# of the three 128-channel MemBlocks.  The values are derived from the pinned
# TAEHV shapes, not from an allocator observation.
DECODER_GROW_WEIGHT_BYTES_BF16 = 720_896
DECODER_HISTORY_MEMBLOCK_WEIGHT_BYTES_BF16 = 1_180_416
DECODER_STAGED_LATENTS = 5
DECODER_STAGED_GROW_CHILDREN = 2
DECODER_STAGED_FRAME_GROUPS = 2
DECODER_STAGED_FRAME_CHILDREN = 1
DECODER_STAGED_DISK_RESERVE_BYTES = 16 << 30
DECODER_STAGED_HOST_RESERVE_BYTES = 2 << 30
DECODER_STAGED_DEVICE_RESERVE_BYTES = 2 << 30
# The fused graph's observed B580 allocator delta is ~4.98 GiB beyond its
# named buffers.  Keep the historical 4 GiB caller minimum, but never report
# less than this measured floor for the fused group.
DECODER_FUSED_MIN_EXTRA_DEVICE_BYTES = 5 << 30
POST_CROSS_BRIDGE_RESIDENT_BANKS = 25  # 24 permanent banks plus one refill bank
POST_CROSS_BRIDGE_WORKSPACE_BYTES = 188_749_824
DECODER_PHASE_ALIASES = {
    "front": "front",
    "history": "history",
    "history0": "history",
    "history1": "history",
    "frame": "frame",
    "frame0": "frame",
    "frame1": "frame",
}
MAX_INT_BYTES = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class MemoryItem:
    name: str
    bytes: int
    note: str


@dataclass(frozen=True, slots=True)
class PlusMemoryPlan:
    geometry: FrameGeometry
    device_items: tuple[MemoryItem, ...]
    host_items: tuple[MemoryItem, ...]
    projection_items: tuple[MemoryItem, ...]
    decoder_items: tuple[MemoryItem, ...]
    decoder_host_items: tuple[MemoryItem, ...]
    available_device_bytes: int
    available_host_bytes: int
    reserve_device_bytes: int
    reserve_host_bytes: int
    required_device_bytes: int
    required_host_bytes: int
    projection_phase: bool
    decoder_phase: bool
    decoder_remote_state: bool
    phase_grouping: str
    phase_lifetimes_disjoint: bool
    extra_buffers_complete: bool
    assumptions: tuple[str, ...]
    host_phase_grouping: str = "sum"
    host_phase_peaks: tuple[MemoryItem, ...] = ()
    remote_tokens: bool = False
    post_cross_bridge: bool = False
    post_ffn2_state: bool = False
    mode: str = "turbo"
    disk_history: bool = False
    prefetched_history: bool = False
    history_disk_bytes: int = 0

    @property
    def device_fits(self) -> bool:
        return self.required_device_bytes <= self.available_device_bytes - self.reserve_device_bytes

    @property
    def host_fits(self) -> bool:
        return self.required_host_bytes <= self.available_host_bytes - self.reserve_host_bytes

    @property
    def fits(self) -> bool:
        # A budget comparison is not an admission proof until the controller
        # has accounted for intentionally unmodelled phase buffers.
        return self.extra_buffers_complete and self.device_fits and self.host_fits

    @property
    def item_bytes(self) -> dict[str, int]:
        return {
            item.name: item.bytes
            for item in (*self.device_items, *self.host_items,
                         *self.projection_items, *self.decoder_items,
                         *self.decoder_host_items)
        }


@dataclass(frozen=True, slots=True)
class DecoderPhaseMemory:
    """Named BF16 buffers for one sequential phased-decoder graph.

    ``device_items`` intentionally contains only graph-owned named buffers and
    active BF16 constants.  Compiler workspaces and allocator fragmentation
    remain explicit allowances; this is an inventory, never a peak proof.
    ``host_items`` includes the canonical nine BF16 histories, not a second
    F32 copy of them.  A parent middle is retained as a remote or host BF16
    tensor when the preceding graph is retired before the next graph compiles.
    """

    geometry: FrameGeometry
    phase: str
    device_items: tuple[MemoryItem, ...]
    host_items: tuple[MemoryItem, ...]
    extra_device_bytes: int
    extra_host_bytes: int
    latent_count: int
    parent_storage: str

    @property
    def required_device_bytes(self) -> int:
        return _add([item.bytes for item in self.device_items], "phased decoder device inventory")

    @property
    def required_host_bytes(self) -> int:
        return _add([item.bytes for item in self.host_items], "phased decoder host inventory")

    @property
    def item_bytes(self) -> dict[str, int]:
        return {item.name: item.bytes for item in (*self.device_items, *self.host_items)}


@dataclass(frozen=True, slots=True)
class StagedDecoderMemory:
    """Named inventory for one group of the source-ordered staged decoder.

    The default staged controller has seven disjoint groups: ``front``, the
    resident ``grow0/grow1`` pair, one graph for each of
    ``memory17``/``memory18``/``memory19``, then separate ``frame0`` and
    ``frame1`` groups.  The opt-in disk fused-memory ledger replaces the three
    memory groups with one ``memory17_19`` group.  Host parent lists are
    deliberately represented in the group where they are simultaneously live;
    the ``previous_input`` item in a non-fused memory group is one canonical
    owner reused by reference after the first source step, matching the source
    MemBlock contract.
    """

    geometry: FrameGeometry
    phase: str
    device_items: tuple[MemoryItem, ...]
    host_items: tuple[MemoryItem, ...]
    extra_device_bytes: int
    extra_host_bytes: int
    latent_count: int
    resident_children: int

    @property
    def required_device_bytes(self) -> int:
        return _add([item.bytes for item in self.device_items],
                    "staged decoder device inventory")

    @property
    def required_host_bytes(self) -> int:
        return _add([item.bytes for item in self.host_items],
                    "staged decoder host inventory")

    @property
    def item_bytes(self) -> dict[str, int]:
        return {item.name: item.bytes for item in (*self.device_items, *self.host_items)}

    @property
    def parent_storage(self) -> str:
        return "host"

    @property
    def source_frames(self) -> int | None:
        return None

    @property
    def output_frames(self) -> int | None:
        return None

    @property
    def history_parent_count(self) -> int:
        return 2 * self.latent_count

    @property
    def disk_storage_bytes(self) -> int:
        return 0

    @property
    def disk_io_bytes(self) -> int:
        return 0

    @property
    def disk_reserve_bytes(self) -> int:
        return 0

    @property
    def reserve_device_bytes(self) -> int:
        return 0

    @property
    def reserve_host_bytes(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class _DiskStagedDecoderMemory(StagedDecoderMemory):
    """Disk-plan metadata kept separate from the legacy host inventory."""

    source_frames: int = field()
    output_frames: int = field()
    disk_items: tuple[MemoryItem, ...] = field()

    @property
    def parent_storage(self) -> str:
        return "disk"

    @property
    def item_bytes(self) -> dict[str, int]:
        return {
            item.name: item.bytes
            for item in (*self.device_items, *self.host_items, *self.disk_items)
        }

    @property
    def disk_storage_bytes(self) -> int:
        return next((item.bytes for item in self.disk_items
                     if item.name.endswith("disk_storage_bytes")), 0)

    @property
    def disk_io_bytes(self) -> int:
        return next((item.bytes for item in self.disk_items
                     if item.name.endswith("disk_io_bytes")), 0)

    @property
    def disk_reserve_bytes(self) -> int:
        return next((item.bytes for item in self.disk_items
                     if item.name.endswith("disk_reserve_bytes")), 0)

    @property
    def reserve_device_bytes(self) -> int:
        return DECODER_STAGED_DEVICE_RESERVE_BYTES

    @property
    def reserve_host_bytes(self) -> int:
        return DECODER_STAGED_HOST_RESERVE_BYTES


def _int(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or (value <= 0 if positive else value < 0):
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(f"{name} must be a {qualifier}integer")
    return value


def _mul(*values, name="memory product"):
    result = 1
    for value in values:
        result *= _int(value, name, positive=False)
        if result > MAX_INT_BYTES:
            raise OverflowError(f"{name} exceeds signed 63-bit byte bound")
    return result


def _add(values, name="memory total"):
    result = 0
    for value in values:
        result += _int(value, name, positive=False)
        if result > MAX_INT_BYTES:
            raise OverflowError(f"{name} exceeds signed 63-bit byte bound")
    return result


def _weight_elements() -> tuple[int, int, int]:
    # Header-only access keeps admission independent of the 5.7 GB payload.
    import plus_weights

    schema = plus_weights.EXPECTED_SCHEMA
    if len(schema) != 825 or len(set(schema)) != 825:
        raise ValueError("Plus DiT schema must contain exactly 825 tensors")
    total = sum(math.prod(shape) for shape in schema.values())
    global_elements = sum(math.prod(shape) for shape in plus_weights.GLOBAL_SHAPES.values())
    layer_elements = sum(math.prod(shape) for shape in plus_weights.LAYER_SHAPES.values())
    if total != global_elements + 30 * layer_elements:
        raise ValueError("Plus DiT schema weight partition changed")
    return total, global_elements, layer_elements


def _projection_elements() -> int:
    """Return the eight released LQ projection tensor elements, header-only."""

    import plus_projection

    shapes = plus_projection.WEIGHT_SHAPES
    if set(shapes) != set(plus_projection.WEIGHT_KEYS) or len(shapes) != 8:
        raise ValueError("Plus LQ projection schema must contain exactly 8 tensors")
    return _add(
        [math.prod(shapes[key]) for key in plus_projection.WEIGHT_KEYS],
        "LQ projection weights",
    )


def _item(name, bytes_, note):
    bytes_ = _int(bytes_, name)
    if bytes_ > MAX_INT_BYTES:
        raise OverflowError(f"{name} exceeds signed 63-bit byte bound")
    return MemoryItem(name, bytes_, note)


def _stream_decoder_plan(geometry, output_frames, stream_call_count):
    """Validate the authenticated fixed-four-latent decoder schedule."""

    if type(output_frames) is not int:
        raise ValueError("decoder_output_frames must be an integer")
    if not DECODER_STREAM_MIN_FRAMES <= output_frames <= DECODER_STREAM_MAX_FRAMES:
        raise ValueError(
            "decoder_output_frames must be in [17, 49] for streaming decode"
        )
    if max(geometry.working_hw) > DECODER_STREAM_MAX_WORKING_SIDE:
        raise ValueError("streaming decoder remains limited to padded HD")
    from plus_stream import build_stream_plan

    plan = build_stream_plan(output_frames, fps=1.0)
    if plan.call_count != stream_call_count:
        raise ValueError(
            "decoder_output_frames does not match stream_call_count"
        )
    return plan


def _budget(value, name):
    value = _int(value, name)
    if value > MAX_INT_BYTES:
        raise OverflowError(f"{name} exceeds signed 63-bit byte bound")
    return value


def decoder_phase_memory(
    geometry: FrameGeometry,
    phase: str,
    *,
    latent_count: int = 1,
    parent_storage: str = "remote",
    retain_parent: bool = True,
    retain_history: bool = True,
    extra_device_bytes: int = 0,
    extra_host_bytes: int = 0,
) -> DecoderPhaseMemory:
    """Inventory one source-ordered phase under compile/retire ownership.

    The intended lifecycle is ``front -> history{0,1} -> frame{0,1}``:
    retire a phase graph after its inference while retaining only its remote
    ``middle`` parent (and canonical host BF16 states) for the next phase.
    ``latent_count`` is the number of four-frame latent windows staged through
    the phase before its graph is retired. ``history0``/``history1`` and
    ``frame0``/``frame1`` are aliases because
    their named buffer geometry is identical; each child has its own half of
    the corresponding TGrow weights.  This does not model OpenVINO allocator
    peaks, so callers must provide explicit extra allowances. ``parent_storage``
    may be ``"remote"`` for same-context handoff or ``"host"`` for a
    compile/retire lifecycle; host handoff is BF16 storage, never host F32.
    """
    if not isinstance(geometry, FrameGeometry):
        raise TypeError("geometry must be a FrameGeometry")
    if not isinstance(phase, str) or phase not in DECODER_PHASE_ALIASES:
        raise ValueError(f"unknown decoder phase: {phase!r}")
    if not isinstance(retain_parent, bool) or not isinstance(retain_history, bool):
        raise ValueError("phase retention flags must be boolean")
    latent_count = _int(latent_count, "latent_count", positive=True)
    if parent_storage not in ("remote", "host"):
        raise ValueError("parent_storage must be 'remote' or 'host'")
    extra_device_bytes = _budget(extra_device_bytes, "extra_device_bytes")
    extra_host_bytes = _budget(extra_host_bytes, "extra_host_bytes")
    phase = DECODER_PHASE_ALIASES[phase]
    h, w = geometry.working_hw
    storage = 2

    state_values = (
        3 * 512 * (h // 8) * (w // 8)
        + 3 * 256 * (h // 4) * (w // 4)
        + 3 * 128 * (h // 2) * (w // 2)
    )
    host = [
        _item("decoder_phase_state_host_bf16", _mul(state_values, storage,
              name="phased decoder host states"),
              "nine canonical state arrays; no parallel host-F32 state copy"),
        _item("decoder_phase_extra_host_allowance", extra_host_bytes,
              "source/packed/readback staging supplied by the controller"),
    ]
    device = [
        _item("decoder_phase_active_weights_bf16",
              DECODER_PHASE_WEIGHT_BYTES_BF16[phase],
              "source decoder constants for this graph only"),
        _item("decoder_phase_extra_device_allowance", extra_device_bytes,
              "OpenVINO graph workspaces, uploads, and allocator headroom"),
    ]

    middle_front = _mul(latent_count, 256, h // 4, w // 4, storage,
                        name="front middle retention")
    middle_history = _mul(2 * latent_count, 128, h // 2, w // 2, storage,
                          name="history middle retention")
    if phase == "front":
        device.extend((
            _item("decoder_front_private_results_bf16", _mul(
                3 * 512 * (h // 8) * (w // 8)
                + 3 * 256 * (h // 4) * (w // 4), storage,
                name="front private Results"),
                "six front Conv residual Results remain remote"),
            _item("decoder_front_middle_bf16", _mul(
                256, h // 4, w // 4, storage, name="front middle"),
                "one active remote output while front runs"),
        ))
        if parent_storage == "host":
            host.append(_item("decoder_front_middle_retained_host_bf16", middle_front,
                              "all front parents retained across graph retirement"))
        elif latent_count > 1:
            device.append(_item("decoder_front_middle_retained_bf16", middle_front,
                                "all front parents retained in the same context"))
    elif phase == "history":
        device.extend((
            _item("decoder_history_private_results_bf16", _mul(
                3 * 128, h // 2, w // 2, storage,
                name="history private Results"),
                "three history Conv residual Results remain remote"),
            _item("decoder_history_middle_bf16", _mul(
                128, h // 2, w // 2, storage, name="history middle"),
                "remote parent retained for the frame phase"),
        ))
        if retain_parent:
            if parent_storage == "host":
                host.append(_item("decoder_front_middle_retained_host_bf16", middle_front,
                                  "front parents supplied from host BF16"))
            else:
                device.append(_item("decoder_front_middle_retained_bf16", middle_front,
                                    "front graph retired; remote parents remain live"))
        if retain_history:
            if parent_storage == "host":
                host.append(_item("decoder_history_middle_retained_host_bf16", middle_history,
                                  "history child parents retained for the frame phase"))
            else:
                device.append(_item("decoder_history_middle_retained_bf16", middle_history,
                                    "history child parents retained for the frame phase"))
    else:
        if retain_history:
            if parent_storage == "host":
                host.append(_item("decoder_history_middle_retained_host_bf16", middle_history,
                                  "both history child parents retained in host BF16"))
            else:
                device.append(_item("decoder_history_middle_retained_bf16", middle_history,
                                    "history graph retired; remote parents remain live"))
        host.append(_item("decoder_frame_output_host_bf16", _mul(
            3, h, w, storage, name="frame output"),
            "public frame Result is bound to host BF16"))

    return DecoderPhaseMemory(
        geometry=geometry,
        phase=phase,
        device_items=tuple(device),
        host_items=tuple(host),
        extra_device_bytes=extra_device_bytes,
        extra_host_bytes=extra_host_bytes,
        latent_count=latent_count,
        parent_storage=parent_storage,
    )


def decoder_sequential_memory(
    geometry: FrameGeometry,
    *,
    latent_count: int = DECODER_CHUNKS * 4,
    extra_device_bytes: int = 0,
    extra_host_bytes: int = 0,
) -> tuple[DecoderPhaseMemory, ...]:
    """Return the three named phase ledgers for compile/retire scheduling.

    The decoder worker has eight four-frame windows for the two-call, 17-frame
    stream.  Staging all front parents and both history-child parents in host
    BF16 permits retiring each graph before compiling the next one; only one
    phase's private Result slabs and learned BF16 constants are device-live.
    """
    latent_count = _int(latent_count, "latent_count", positive=True)
    return tuple(
        decoder_phase_memory(
            geometry, phase, latent_count=latent_count, parent_storage="host",
            extra_device_bytes=extra_device_bytes, extra_host_bytes=extra_host_bytes,
        )
        for phase in ("front", "history", "frame")
    )


def staged_decoder_memory(
    geometry: FrameGeometry,
    *,
    latent_count: int = DECODER_STAGED_LATENTS,
    extra_device_bytes: int = 0,
    extra_host_bytes: int = 0,
    parent_storage: str = "host",
    source_frames: int | None = None,
    output_frames: int | None = None,
    disk_reserve_bytes: int = DECODER_STAGED_DISK_RESERVE_BYTES,
    fuse_memories: bool = False,
    frame_pair: bool = False,
    remote_history: bool = False,
    fuse_growth: bool = False,
    remote_frames: bool = False,
    rgb_source_shape: tuple[int, int, int, int] | None = None,
) -> tuple[StagedDecoderMemory, ...]:
    """Inventory the source-ordered staged decoder groups.

    The default returned plans are ``front``, ``grow``, ``memory17``,
    ``memory18``, ``memory19``, ``frame0``, and ``frame1``.  ``latent_count`` active
    (requested) front parents produce ``2 * latent_count`` chronological
    history currents.  The three
    memory plans intentionally contain one
    MemBlock each; their ten-element host input and output lists are both live,
    while ``decoder_staged_history_previous_input_alias_bf16`` is one canonical
    preceding-input owner that is reused by reference after the first step,
    rather than a copied state tensor per parent.  Grow
    explicitly accounts for its two resident child graphs; frame0 and frame1
    are separate one-child groups so their private device inventories are not
    incorrectly doubled.

    This is a named-buffer inventory, not an OpenVINO allocator peak proof.
    ``extra_*`` are per-active-group allowances for compiler workspaces and
    typed host/device staging owned by the controller.  The opt-in
    ``parent_storage="disk"`` branch validates the canonical fixed-four-latent
    stream plan, keeps only bounded current/past/fresh BF16 parent buffers, and
    reports two alternating BF16 parent files plus an explicit disk reserve.
    Padded latent chunks are accounted separately for the denoised-latent host
    owner; the padded tail is not treated as active history.
    The historical host branch is unchanged.  The strict opt-in
    ``fuse_memories=True`` form is currently admitted only with
    ``parent_storage="disk"``; it returns ``front``, ``grow``,
    ``memory17_19``, ``frame0``, and ``frame1``.  Its fused steady group uses
    the audited three-MemBlock weight set, eight middle-sized BF16 device
    buffers, and the caller's existing workspace allowance.  Its disk I/O
    item is a reportable source-order assumption, not an allocator or timing
    framework.
    ``frame_pair=True`` is a strict disk-only opt-in that combines the two
    stateless frame children; geometry-dependent tile bytes come only from
    ``frame_geometry.frame_pair_tiles(geometry.working_hw)``.
    ``remote_history=True`` is a strict disk/fused opt-in for the
    ``memory17_19`` group.  It accounts a bounded BF16 history ring and a
    separate finite-check workspace; all other groups and disk items remain
    unchanged. ``fuse_growth=True`` additionally folds growth into that ring,
    with eight lanes, both child weight banks, compact input and overlapping
    growth intermediates charged explicitly. It removes the separate grow
    group and intermediate disk I/O, not any reserve or workspace allowance.
    ``remote_frames=True`` is a stricter disk/fused opt-in that requires fused
    growth and no frame-pair phase. It keeps the growth graph and one dynamic
    frame-child graph live together, charges both dynamic bank sets and
    eight full-halo BF16 frame-branch intermediates,
    and returns one ``grow_frames`` group with direct normalized CPU outputs.
    It has no history-parent disk files or disk I/O; the existing disk branch is
    retained only for authenticated stream-plan validation.
    ``rgb_source_shape=(F,H,W,3)`` is a disk-only decoder opt-in. It replaces
    only the front group's full normalized-F32 source owner with bounded raw,
    resized, padded, ordering, and BF16 lookup owners; all device, parent-file,
    denoised-chunk, reserve, and workspace items stay unchanged.
    """
    if not isinstance(geometry, FrameGeometry):
        raise TypeError("geometry must be a FrameGeometry")
    if type(fuse_memories) is not bool:
        raise ValueError("fuse_memories must be boolean")
    if type(frame_pair) is not bool:
        raise ValueError("frame_pair must be boolean")
    if type(remote_history) is not bool:
        raise ValueError("remote_history must be boolean")
    if type(fuse_growth) is not bool or (fuse_growth and not remote_history):
        raise ValueError("fuse_growth requires remote_history=True")
    if type(remote_frames) is not bool:
        raise ValueError("remote_frames must be boolean")
    if remote_frames and not fuse_growth:
        raise ValueError("remote_frames requires fuse_growth=True")
    if remote_frames and frame_pair:
        raise ValueError("remote_frames requires frame_pair=False")
    if rgb_source_shape is not None:
        if parent_storage != "disk":
            raise ValueError("rgb_source_shape requires parent_storage='disk'")
        if (not isinstance(rgb_source_shape, tuple)
                or len(rgb_source_shape) != 4
                or any(type(value) is not int or value <= 0
                       for value in rgb_source_shape)
                or rgb_source_shape[3] != 3):
            raise ValueError(
                "rgb_source_shape must be a tuple (frames,height,width,3) of positive integers"
            )
        rgb_frames, rgb_height, rgb_width, _ = rgb_source_shape
        if (rgb_frames > 49 or max(rgb_height, rgb_width) > 3840
                or rgb_height * rgb_width > 2176 * 3840):
            raise ValueError("rgb_source_shape exceeds the admitted source geometry")
    latent_count = _int(latent_count, "latent_count", positive=True)
    extra_device_bytes = _budget(extra_device_bytes, "extra_device_bytes")
    extra_host_bytes = _budget(extra_host_bytes, "extra_host_bytes")
    if parent_storage not in ("host", "disk"):
        raise ValueError("parent_storage must be 'host' or 'disk'")
    if frame_pair and parent_storage != "disk":
        raise ValueError("frame_pair requires parent_storage='disk'")
    if fuse_memories and parent_storage != "disk":
        raise ValueError("fuse_memories requires parent_storage='disk'")
    if remote_history and parent_storage != "disk":
        raise ValueError("remote_history requires parent_storage='disk'")
    if remote_history and not fuse_memories:
        raise ValueError("remote_history requires fuse_memories=True")
    if remote_frames and not remote_history:
        raise ValueError("remote_frames requires remote_history=True")
    if fuse_memories and (
        extra_device_bytes < (4 << 30) or extra_host_bytes < (6 << 30)
    ):
        raise ValueError(
            "fused memory requires the existing 4 GiB device and 6 GiB host allowances"
        )
    padded_latent_count = latent_count
    if parent_storage == "host":
        if source_frames is not None or output_frames is not None:
            raise ValueError("source/output frames require parent_storage='disk'")
        disk_reserve_bytes = _budget(disk_reserve_bytes, "disk_reserve_bytes")
    else:
        if source_frames is None or output_frames is None:
            raise ValueError(
                "disk parent storage requires source_frames and output_frames"
            )
        source_frames = _int(source_frames, "source_frames", positive=True)
        output_frames = _int(output_frames, "output_frames", positive=True)
        if not DECODER_STREAM_MIN_FRAMES <= output_frames <= DECODER_STREAM_MAX_FRAMES:
            raise ValueError("output_frames is outside the reviewed stream range")
        from plus_stream import build_stream_plan
        stream_plan = build_stream_plan(output_frames, fps=1.0)
        if source_frames != stream_plan.projection_frames:
            raise ValueError(
                "source_frames must equal the stream projection frame count"
            )
        if latent_count != stream_plan.requested_latents:
            raise ValueError(
                "latent_count must equal the stream requested latent count"
            )
        padded_latent_count = stream_plan.padded_latents
        disk_reserve_bytes = _budget(disk_reserve_bytes, "disk_reserve_bytes")
        if disk_reserve_bytes < DECODER_STAGED_DISK_RESERVE_BYTES:
            raise ValueError("disk reserve must be at least 16 GiB")
        if rgb_source_shape is not None and rgb_source_shape[0] < output_frames:
            raise ValueError(
                "rgb_source_shape must contain at least the requested output frame count"
            )
    h, w = geometry.working_hw
    storage = 2

    front_middle = _mul(256, h // 4, w // 4, storage,
                        name="staged front middle")
    history_middle = _mul(128, h // 2, w // 2, storage,
                          name="staged history middle")
    packed = _mul(784, h // 8, w // 8, storage,
                  name="staged packed input")
    front_private = _mul(
        3 * 512 * (h // 8) * (w // 8)
        + 3 * 256 * (h // 4) * (w // 4), storage,
        name="staged front private results",
    )
    front_states = _mul(
        3 * 512 * (h // 8) * (w // 8)
        + 3 * 256 * (h // 4) * (w // 4), storage,
        name="staged front states",
    )
    frame_output = _mul(3, h, w, storage, name="staged frame output")
    frame_output_f32 = _mul(3, h, w, 4, name="staged frame output F32")

    history_parent_count = _mul(2, latent_count, name="staged history parent count")
    disk_items = ()
    frame_pair_sizes = None
    if parent_storage == "disk" and frame_pair:
        descriptor = frame_pair_tiles(geometry.working_hw)
        try:
            tile_h, tile_w = tuple(descriptor["tile_hw"])
            middle_shape = tuple(descriptor["middle_shape"])
            if len(middle_shape) != 4 or middle_shape[:2] != (1, 128):
                raise ValueError("unexpected middle_shape")
            middle_h, middle_w = middle_shape[2:]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frame_pair_tiles descriptor ABI changed") from exc
        for name, values in (("tile_hw", (tile_h, tile_w)),
                             ("middle_shape", (middle_h, middle_w))):
            if (len(values) != 2
                    or any(type(value) is not int or value <= 0 for value in values)):
                raise ValueError(
                    f"frame_pair_tiles {name} must be positive integer coordinates")
        tile_input = _mul(128, middle_h, middle_w, storage,
                          name="frame-pair tile input")
        interpolation = _mul(128, tile_h, tile_w, storage,
                             name="frame-pair block20 interpolation")
        pair_outputs = _mul(2, 3, tile_h, tile_w, storage,
                            name="frame-pair tile outputs")
        full_raw = _mul(2, 3, h, w, storage,
                        name="two full raw frame owners")
        normalized = _mul(3, h, w, 4,
                          name="one normalized frame scratch")
        pair_reads = _mul(2, history_parent_count - 1, tile_input,
                          name="frame-pair parent reads")
        baseline_factor = 1 if fuse_growth else 3 if fuse_memories else 7
        baseline_io = _mul(baseline_factor, history_parent_count, history_middle,
                           name="staged grow/memory disk I/O")
        frame_pair_sizes = {
            "tile_input": tile_input,
            "interpolation": interpolation,
            "pair_outputs": pair_outputs,
            "full_raw": full_raw,
            "normalized": normalized,
            "disk_io": _add((baseline_io, pair_reads),
                             "frame-pair disk I/O"),
        }
    if parent_storage == "disk" and not remote_frames:
        disk_file_bytes = _mul(
            history_parent_count, history_middle,
            name="one BF16 staged parent file",
        )
        disk_items = (
            _item(
                "decoder_staged_parent_files_disk_storage_bytes",
                _mul(2, disk_file_bytes, name="two BF16 staged parent files"),
                "two alternating BF16 files; one old and one fresh stage",
            ),
            _item(
                "decoder_staged_parent_files_disk_reserve_bytes",
                disk_reserve_bytes,
                "free disk headroom required in addition to both parent files",
            ),
        )
        if frame_pair:
            disk_items += (_item(
                "decoder_staged_parent_files_disk_io_bytes",
                frame_pair_sizes["disk_io"],
                "one final write when growth is fused, otherwise grow/memory I/O, plus two frame-pair reads for every "
                "non-startup parent pair",
            ),)
        elif fuse_memories:
            disk_items += (_item(
                "decoder_staged_parent_files_disk_io_bytes",
                _mul(3 if fuse_growth else 5, history_parent_count, history_middle,
                     name="fused staged parent disk I/O"),
                "final history write and two frame reads; separate growth also "
                "writes and rereads an intermediate parent",
            ),)
    elif parent_storage == "disk" and remote_frames:
        # Remote-frame mode has no parent files or traffic, but the caller's
        # authenticated scratch admission still requires the fixed reserve.
        disk_items = (_item(
            "decoder_staged_parent_files_disk_reserve_bytes",
            disk_reserve_bytes,
            "required free scratch reserve; no parent files or disk I/O in remote-frame mode",
        ),)
    if parent_storage == "disk":
        # These owners remain live in remote-frame mode even though its
        # history transport has no parent files or disk traffic.
        model_owners = _mul(
            4, DECODER_PARAMETER_BYTES_F32,
            name="staged decoder F32 module/model owners",
        )
        source_video = _mul(
            3, source_frames, h, w, 4,
            name="staged decoder projection source",
        )
        denoised_latents = _mul(
            padded_latent_count, 16, h // 8, w // 8, storage,
            name="staged decoder denoised latent chunks",
        )
    else:
        model_owners = source_video = denoised_latents = 0

    if parent_storage != "disk":
        source_host_items = ()
    elif rgb_source_shape is None:
        source_host_items = (_item(
            "decoder_staged_source_video_host_f32", source_video,
            "authenticated projection source retained through front packing only",
        ),)
    else:
        rgb_frames, rgb_height, rgb_width, _ = rgb_source_shape
        raw_frame_bytes = _mul(
            rgb_height, rgb_width, 3,
            name="one raw RGB source frame",
        )
        visible_frame_bytes = _mul(
            3, geometry.height, geometry.width,
            name="one visible RGB resize frame",
        )
        working_frame_bytes = _mul(
            3, h, w,
            name="one working RGB frame",
        )
        window_bytes = _mul(
            4, working_frame_bytes,
            name="four-frame RGB working window",
        )
        lookup_bytes = _mul(
            768, h // 8, w // 8, storage,
            name="RGB BF16 lookup owner",
        )
        source_host_items = (
            _item(
                "decoder_staged_source_rgb_mapped_u8",
                _mul(rgb_frames, raw_frame_bytes,
                     name="mapped RGB source bytes"),
                "conservative page-residency bound for the authenticated read-only RGB memmap",
            ),
            _item(
                "decoder_staged_source_rgb_raw_frame_copies_u8",
                _mul(2, raw_frame_bytes,
                     name="raw RGB frame copy owners"),
                "two immutable raw-frame copies used for hash-before-resize",
            ),
            _item(
                "decoder_staged_source_rgb_visible_resize_u8",
                _mul(2, visible_frame_bytes,
                     name="visible RGB resize owners"),
                "two visible RGB resize owners before edge padding",
            ),
            _item(
                "decoder_staged_source_rgb_working_frame_u8",
                _mul(2, working_frame_bytes,
                     name="working RGB frame owners"),
                "two working padded RGB frame owners",
            ),
            _item(
                "decoder_staged_source_rgb_window_u8",
                window_bytes,
                "current four-frame working RGB uint8 window",
            ),
            _item(
                "decoder_staged_source_rgb_ordering_u8",
                window_bytes,
                "contiguous uint8 decoder-order staging before table lookup",
            ),
            _item(
                "decoder_staged_source_rgb_lookup_bf16",
                lookup_bytes,
                "one uint16 BF16 lookup result before packed latent assembly",
            ),
        )

    def plan(phase, device, host, children, *, phase_extra_device_bytes=None):
        phase_extra_device_bytes = (
            extra_device_bytes if phase_extra_device_bytes is None
            else _budget(phase_extra_device_bytes, "phase_extra_device_bytes")
        )
        if parent_storage == "disk":
            if phase != "front":
                host = (*host, _item(
                    "decoder_staged_denoised_latents_host_bf16", denoised_latents,
                    "caller-owned BF16 chunks retained for post-clock diagnostic archives"))
            return _DiskStagedDecoderMemory(
                geometry=geometry, phase=phase,
                device_items=tuple(device), host_items=tuple(host),
                extra_device_bytes=phase_extra_device_bytes,
                extra_host_bytes=extra_host_bytes,
                latent_count=latent_count, resident_children=children,
                source_frames=source_frames, output_frames=output_frames,
                disk_items=tuple(disk_items),
            )
        return StagedDecoderMemory(
            geometry=geometry, phase=phase,
            device_items=tuple(device), host_items=tuple(host),
            extra_device_bytes=phase_extra_device_bytes,
            extra_host_bytes=extra_host_bytes,
            latent_count=latent_count, resident_children=children,
        )

    front = plan(
        "front",
        (
            _item("decoder_staged_front_weights_bf16",
                  DECODER_PHASE_WEIGHT_BYTES_BF16["front"],
                  "one front graph's source F32 weights stored as BF16"),
            _item("decoder_staged_front_packed_input_bf16", packed,
                  "one typed packed latent input"),
            _item("decoder_staged_front_private_results_bf16", front_private,
                  "six source front MemBlock private Results"),
            _item("decoder_staged_front_middle_output_bf16",
                  front_middle, "one remote front middle output"),
            _item("decoder_staged_front_workspace_allowance",
                  extra_device_bytes,
                  "compiler workspace and device upload allowance"),
        ),
        (
            _item("decoder_staged_front_state_host_bf16", front_states,
                  "six canonical front state output arrays"),
            _item("decoder_staged_front_parent_list_host_bf16",
                  _mul(latent_count, front_middle,
                       name="staged front parent list"),
                  "front middle parents retained for both growth children"),
            _item("decoder_staged_front_typed_io_allowance",
                  extra_host_bytes,
                  "typed host input/output staging owned by the controller"),
            *(() if parent_storage == "host" else (
                _item("decoder_staged_decoder_model_owners_host_f32", model_owners,
                      "module, checkpoint/model clones retained during staged graph setup"),
                *source_host_items,
                _item("decoder_staged_denoised_latents_host_bf16", denoised_latents,
                      "caller-owned fixed-four-latent BF16 chunks consumed by front"),
                _item("decoder_staged_front_state_previous_host_bf16", front_states,
                      "previous front state arrays live during one handoff"),
                _item("decoder_staged_front_state_typed_input_host_bf16", front_states,
                      "typed front state feeds created from raw storage bits"),
                _item("decoder_staged_front_state_output_tensor_host_bf16", front_states,
                      "host-bound graph output tensors"),
                _item("decoder_staged_front_middle_readback_host_bf16",
                      _mul(2, front_middle, name="front middle host readback"),
                      "host Tensor plus returned BF16 array before parent retention"),
                _item("decoder_staged_front_packed_typed_host_bf16",
                      _mul(2, packed, name="front packed host feed"),
                      "raw packed bits plus typed input owner"),
            )),
        ),
        1,
    )

    grow = plan(
        "grow",
        (
            _item("decoder_staged_grow_weights_bf16",
                  _mul(DECODER_STAGED_GROW_CHILDREN,
                       DECODER_GROW_WEIGHT_BYTES_BF16,
                       name="staged grow child weights"),
                  "grow0 and grow1 are resident together"),
            _item("decoder_staged_grow_parent_input_bf16", front_middle,
                  "one front parent input"),
            _item("decoder_staged_grow_middle_output_bf16",
                  _mul(DECODER_STAGED_GROW_CHILDREN, history_middle,
                       name="staged grow middle outputs"),
                  "one history current output per resident grow child"),
            _item("decoder_staged_grow_workspace_allowance",
                  extra_device_bytes,
                  "compiler workspace and device upload allowance"),
        ),
        (
            _item("decoder_staged_grow_parent_list_host_bf16",
                  _mul(latent_count, front_middle,
                       name="staged grow parent list"),
                  "front parents remain live while both growth children run"),
            *(
                (_item("decoder_staged_history_input_parent_list_host_bf16",
                       _mul(2 * latent_count, history_middle,
                            name="staged history input parent list"),
                       "ten grow outputs retained in source chronology"),)
                if parent_storage == "host" else (
                    _item("decoder_staged_decoder_model_owners_host_f32", model_owners,
                          "module, checkpoint/model clones retained during staged graph setup"),
                    _item("decoder_staged_grow_parent_input_host_bf16",
                          _mul(2, front_middle, name="grow parent typed handoff"),
                          "front parent raw array plus typed graph feed"),
                    _item("decoder_staged_grow_fresh_result_host_bf16",
                          _mul(2, history_middle, name="grow result handoff"),
                          "graph result host Tensor plus returned BF16 array before disk write"),
                )
            ),
            _item("decoder_staged_grow_typed_io_allowance", extra_host_bytes,
                  "typed host input/output staging owned by the controller"),
        ),
        DECODER_STAGED_GROW_CHILDREN,
    )

    history = []
    for index in (() if fuse_memories else (17, 18, 19)):
        history.append(plan(
            f"memory{index}",
            (
                _item(f"decoder_staged_memory{index}_weights_bf16",
                      DECODER_HISTORY_MEMBLOCK_WEIGHT_BYTES_BF16,
                      "one source 128-channel MemBlock"),
                _item(f"decoder_staged_memory{index}_current_input_allowance_bf16",
                      history_middle, "one explicit current GPU-input allowance"),
                _item(f"decoder_staged_memory{index}_past_input_allowance_bf16",
                      history_middle, "one explicit preceding-input GPU allowance"),
                _item(f"decoder_staged_memory{index}_private_result_bf16",
                      history_middle, "one private Conv Result"),
                _item(f"decoder_staged_memory{index}_middle_output_bf16",
                      history_middle, "one BF16 middle output"),
                _item(f"decoder_staged_memory{index}_workspace_allowance",
                      extra_device_bytes,
                      "compiler workspace and device upload allowance"),
            ),
            (
                *(
                    (
                        _item(f"decoder_staged_memory{index}_input_parent_list_host_bf16",
                              _mul(2 * latent_count, history_middle,
                                   name="staged memory input parent list"),
                              "ten chronological current inputs"),
                        _item(f"decoder_staged_memory{index}_output_parent_list_host_bf16",
                              _mul(2 * latent_count, history_middle,
                                   name="staged memory output parent list"),
                              "ten chronological outputs for the next MemBlock"),
                        _item(f"decoder_staged_memory{index}_previous_input_alias_bf16",
                              history_middle,
                              "one canonical preceding INPUT owner; later steps alias current, not copy"),
                    )
                    if parent_storage == "host" else (
                        _item(f"decoder_staged_memory{index}_decoder_model_owners_host_f32",
                              model_owners,
                              "module, checkpoint/model clones retained during staged graph setup"),
                        _item(f"decoder_staged_memory{index}_current_input_host_bf16",
                              history_middle,
                              "preadv current BF16 parent buffer"),
                        _item(f"decoder_staged_memory{index}_past_input_host_bf16",
                              history_middle,
                              "previous INPUT BF16 buffer, never previous output"),
                        _item(f"decoder_staged_memory{index}_fresh_result_host_bf16",
                              history_middle,
                              "fresh BF16 result buffer written to the alternate file"),
                        _item(f"decoder_staged_memory{index}_current_typed_input_host_bf16",
                              history_middle,
                              "typed current graph feed owner"),
                        _item(f"decoder_staged_memory{index}_past_typed_input_host_bf16",
                              history_middle,
                              "typed past graph feed owner"),
                        _item(f"decoder_staged_memory{index}_result_readback_host_bf16",
                              history_middle,
                              "host Tensor/returned array handoff before disk write"),
                    )
                ),
                _item(f"decoder_staged_memory{index}_typed_io_allowance",
                      extra_host_bytes,
                      "typed current/past input and output staging owned by the controller"),
            ),
            1,
        ))

    if fuse_memories:
        fused_prefix = "decoder_staged_memory17_19"
        fused_extra_device_bytes = max(
            extra_device_bytes, DECODER_FUSED_MIN_EXTRA_DEVICE_BYTES
        )
        history.append(plan(
            "memory17_19",
            (
                _item(f"{fused_prefix}_weights_bf16",
                      _mul(3, DECODER_HISTORY_MEMBLOCK_WEIGHT_BYTES_BF16,
                           name="fused memory block weights"),
                      "three source 128-channel MemBlocks"),
                _item(f"{fused_prefix}_state_abi_bf16",
                      _mul(5, history_middle, name="fused memory state ABI"),
                      "current and three past input allowances plus final middle; "
                      "state18/state19 outputs are host-bound, state17 reuses current"),
                _item(f"{fused_prefix}_private_results_bf16",
                      _mul(3, history_middle, name="fused memory private Results"),
                      "three source MemBlock private BF16 Results"),
                _item(f"{fused_prefix}_workspace_allowance",
                      fused_extra_device_bytes,
                      "compiler workspace and device upload allowance; observed fused floor"),
            ),
            (
                _item(f"{fused_prefix}_decoder_model_owners_host_f32",
                      model_owners,
                      "module, checkpoint/model clones retained during staged graph setup"),
                _item(f"{fused_prefix}_raw_inputs_host_bf16",
                      _mul(4, history_middle, name="fused memory raw inputs"),
                      "four current/past BF16 raw input owners"),
                _item(f"{fused_prefix}_fresh_reads_host_bf16",
                      _mul(2, history_middle, name="fused memory fresh reads"),
                      "two fresh BF16 read buffers"),
                _item(f"{fused_prefix}_typed_inputs_host_bf16",
                      _mul(4, history_middle, name="fused memory typed inputs"),
                      "four typed BF16 graph input owners"),
                _item(f"{fused_prefix}_state_tensors_host_bf16",
                      _mul(6, history_middle, name="fused memory state tensors"),
                      "six state tensors/copies retained across the steady ABI"),
                _item(f"{fused_prefix}_typed_io_allowance",
                      extra_host_bytes,
                      "existing typed host staging allowance"),
            ),
            1,
            phase_extra_device_bytes=fused_extra_device_bytes,
        ))

        if remote_history:
            # The pure geometry descriptor sizes a bounded remote BF16 ring.
            # Count the current input once; S1/S2 host states and diagnostics
            # are intentionally not part of this production ledger.
            descriptor = history_memory_tiles(
                geometry.working_hw,
                **(
                    {"tile_count": 16, "halo_rows": 12}
                    if remote_frames else
                    {"tile_count": 8} if fuse_growth else {}
                ),
            )
            try:
                tile_shape = tuple(descriptor["tile_middle_shape"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("history_memory_tiles descriptor ABI changed") from exc
            if (len(tile_shape) != 4 or tile_shape[:2] != (1, 128)
                    or any(type(value) is not int or value <= 0
                           for value in tile_shape)):
                raise ValueError("history tile shape must be positive [1,128,H,W]")
            history_tile_bytes = _mul(*tile_shape, storage,
                                      name="history tile BF16 bytes")
            tile_source_hw = tuple(descriptor.get("tile_source_hw", ()))
            if (len(tile_source_hw) != 2
                    or any(type(value) is not int or value <= 0
                           for value in tile_source_hw)):
                raise ValueError("history tile source geometry is malformed")
            history_checker_u8 = history_tile_bytes // 2
            compact_tile_bytes = history_tile_bytes // 2
            if history_checker_u8 <= 0:
                raise ValueError("history checker mask is empty")
            frame_interpolation_bytes = _mul(
                128, *tile_source_hw, storage,
                name="remote frame interpolation tile",
            )
            frame_output_bytes = _mul(
                2, 3, *tile_source_hw, storage,
                name="remote frame output tiles",
            )
            frame_full_raw_bytes = _mul(
                2, 3, h, w, storage,
                name="remote full raw frame owners",
            )
            frame_normalized_scratch_bytes = _mul(
                3, h, w, 4,
                name="remote normalized frame scratch",
            )
            frame_final_output_bytes = _mul(
                output_frames, 3, geometry.height, geometry.width, 4,
                name="remote normalized final output",
            )
            frame_dynamic_bank_bytes = _mul(
                2, 128, 128, 1, 1, storage,
                name="remote frame dynamic banks",
            )
            growth_dynamic_bank_bytes = _mul(
                2, 256, 256, 1, 1, storage,
                name="remote growth dynamic banks",
            )
            frame_branch_intermediates = _mul(
                8, history_tile_bytes,
                name="remote frame branch intermediates",
            )
            remote_device = (
                _item(f"{fused_prefix}_weights_bf16",
                      _add((_mul(3, DECODER_HISTORY_MEMBLOCK_WEIGHT_BYTES_BF16,
                                 name="remote history block weights"),
                            DECODER_GROW_WEIGHT_BYTES_BF16 if fuse_growth else 0),
                           "remote history weights"),
                      "three source MemBlocks, plus both growth banks when fused"),
                _item(f"{fused_prefix}_ring_slots_bf16",
                      _mul(3 * (len(descriptor["tiles"]) + 1), history_tile_bytes,
                           name="remote history ring slots"),
                      "three state roots per spatial lane, plus three free/output roots"),
                _item(f"{fused_prefix}_middle_bf16", history_tile_bytes,
                      "one tiled fused history middle output"),
                _item(f"{fused_prefix}_private_results_bf16",
                      _mul(3, history_tile_bytes,
                           name="remote history private Results"),
                      "three source MemBlock private Results"),
                *(() if not fuse_growth else (
                    _item(f"{fused_prefix}_growth_intermediates_bf16",
                          _mul(4, history_tile_bytes, name="growth intermediate overlap"),
                          "block14 and block15 values conservatively live together"),
                    _item(f"{fused_prefix}_compact_input_bf16", compact_tile_bytes,
                          "one separate compact input root; never aliases grown S0"),
                )),
                _item(f"{fused_prefix}_checker_abs_bf16", history_tile_bytes,
                      "one BF16 absolute-value checker input"),
                _item(f"{fused_prefix}_checker_f32",
                      _mul(2, history_tile_bytes,
                           name="remote history checker F32"),
                      "two-byte BF16 tile expanded to F32"),
                _item(f"{fused_prefix}_checker_finite_u8", history_checker_u8,
                      "one byte finite bitmap per BF16 element"),
                _item(f"{fused_prefix}_checker_f16_storage", history_tile_bytes,
                      "one F16 checker storage tile"),
                _item(f"{fused_prefix}_checker_flags", 2,
                      "one F16 scalar finite flag"),
                _item(f"{fused_prefix}_checker_allowance", 1 << 30,
                      "separate one GiB finite-check allocator allowance"),
                _item(f"{fused_prefix}_workspace_allowance",
                      fused_extra_device_bytes,
                      "compiler workspace and device upload allowance; observed fused floor"),
                *(() if not remote_frames else (
                    _item(f"{fused_prefix}_frame_weights_bf16",
                          DECODER_PHASE_WEIGHT_BYTES_BF16["frame"],
                          "one resident dynamic frame-child graph"),
                    _item(f"{fused_prefix}_growth_dynamic_banks_bf16",
                          growth_dynamic_bank_bytes,
                          "both resident BF16 growth-bank roots"),
                    _item(f"{fused_prefix}_frame_dynamic_banks_bf16",
                          frame_dynamic_bank_bytes,
                          "both resident BF16 frame-bank roots"),
                    _item(f"{fused_prefix}_frame_input_bf16",
                          history_tile_bytes,
                          "one dynamic frame-child middle input with full halo"),
                    _item(f"{fused_prefix}_frame_interpolation_bf16",
                          frame_interpolation_bytes,
                          "one full-halo block-20 interpolation result"),
                    _item(f"{fused_prefix}_frame_output_bf16",
                          frame_output_bytes,
                          "two full-halo dynamic frame outputs"),
                    _item(f"{fused_prefix}_frame_branch_intermediates_bf16",
                          frame_branch_intermediates,
                          "eight full-halo BF16 block20/current-child intermediates; not shared with fused history"),
                )),
            )
            remote_host = (
                _item(f"{fused_prefix}_decoder_model_owners_host_f32",
                      model_owners,
                      "module, checkpoint/model clones retained during staged graph setup"),
                _item(f"{fused_prefix}_current_host_bf16",
                      _mul(latent_count, front_middle, name="retained compact parents")
                      if fuse_growth else history_middle,
                      "all compact front parents until fused growth finishes"
                      if fuse_growth else "one full current BF16 history owner"),
                _item(f"{fused_prefix}_result_host_bf16", history_middle,
                      "one full BF16 result owner before tile scatter"),
                _item(f"{fused_prefix}_tile_input_host_bf16",
                      compact_tile_bytes if fuse_growth else history_tile_bytes,
                      "one bounded BF16 history tile input"),
                _item(f"{fused_prefix}_middle_readback_host_bf16",
                      history_tile_bytes,
                      "one bounded host middle readback tile"),
                _item(f"{fused_prefix}_initial_zero_tile_host_bf16",
                      history_tile_bytes,
                      "one conservative initial zero tile"),
                *(() if not fuse_growth else (
                    _item(f"{fused_prefix}_growth_banks_host_bf16", 262_144,
                          "two owning source-rounded BF16 growth banks"),
                )),
                *(() if not remote_frames else (
                    _item(f"{fused_prefix}_frame_dynamic_banks_host_bf16",
                          frame_dynamic_bank_bytes,
                          "two owning source-rounded BF16 frame banks"),
                    _item(f"{fused_prefix}_frame_input_host_bf16",
                          history_tile_bytes,
                          "dynamic frame-child middle input staging"),
                    _item(f"{fused_prefix}_frame_output_tile_host_bf16",
                          frame_output_bytes,
                          "full-halo frame outputs before direct CPU normalization"),
                    _item(f"{fused_prefix}_frame_raw_owners_host_bf16",
                          frame_full_raw_bytes,
                          "conservative two full working-frame BF16 owners"),
                    _item(f"{fused_prefix}_frame_normalized_scratch_host_f32",
                          frame_normalized_scratch_bytes,
                          "conservative one full working-frame normalization scratch"),
                    _item(f"{fused_prefix}_frame_final_output_host_f32",
                          frame_final_output_bytes,
                          "direct normalized CPU output owner"),
                )),
                _item(f"{fused_prefix}_typed_io_allowance", extra_host_bytes,
                      "existing typed host staging allowance"),
            )
            history[-1] = plan(
                "grow_frames" if remote_frames else ("grow_memory" if fuse_growth else "memory17_19"),
                remote_device, remote_host, 1,
                phase_extra_device_bytes=fused_extra_device_bytes,
            )

    # The remote frame controller consumes the fused growth group directly;
    # there are no separate frame0/frame1 groups or parent files in this mode.
    if remote_frames:
        return (front, *history)

    if parent_storage == "disk" and frame_pair:
        prefix = "decoder_staged_frame_pair"
        frame_pair_plan = plan(
            "frame_pair",
            (
                _item(f"{prefix}_weights_bf16",
                      _mul(2, DECODER_PHASE_WEIGHT_BYTES_BF16["frame"],
                           name="frame-pair branch weights"),
                      "both stateless frame branches are resident in one graph"),
                _item(f"{prefix}_tile_input_bf16",
                      frame_pair_sizes["tile_input"],
                      "one tiled middle input"),
                _item(f"{prefix}_block20_interpolation_bf16",
                      frame_pair_sizes["interpolation"],
                      "one shared block-20 interpolation tile"),
                _item(f"{prefix}_output_bf16",
                      frame_pair_sizes["pair_outputs"],
                      "two BF16 tile outputs"),
                _item(f"{prefix}_workspace_allowance",
                      _mul(2, extra_device_bytes,
                           name="two frame-pair workspace allowances"),
                      "one unchanged caller allowance per frame branch"),
            ),
            (
                _item(f"{prefix}_decoder_model_owners_host_f32", model_owners,
                      "module, checkpoint/model clones retained during setup"),
                _item(f"{prefix}_raw_frame_owners_host_bf16",
                      frame_pair_sizes["full_raw"],
                      "two full padded raw BF16 frame owners"),
                _item(f"{prefix}_bound_tile_outputs_host_bf16",
                      frame_pair_sizes["pair_outputs"],
                      "two host-bound BF16 tile outputs"),
                _item(f"{prefix}_tile_input_host_bf16",
                      frame_pair_sizes["tile_input"],
                      "one host tile input owner"),
                _item(f"{prefix}_normalized_frame_scratch_host_f32",
                      frame_pair_sizes["normalized"],
                      "one transient padded F32 normalized frame"),
                _item(f"{prefix}_normalized_output_owner_host_f32",
                      _mul(output_frames, 3, geometry.height, geometry.width, 4,
                           name="staged normalized output owner"),
                      "one visible output owner written per frame"),
                _item(f"{prefix}_normalization_scratch_host_f32",
                      _mul(6, frame_output_f32,
                           name="staged frame-pair normalization scratch"),
                      "existing conservative six-frame F32 RNE scratch"),
                _item(f"{prefix}_typed_io_allowance", extra_host_bytes,
                      "typed host input/output staging owned by the controller"),
            ),
            DECODER_STAGED_FRAME_GROUPS,
            phase_extra_device_bytes=_mul(
                2, extra_device_bytes, name="frame-pair workspace allowance"),
        )
        return (front, *((grow,) if not fuse_growth else ()), *history, frame_pair_plan)

    frames = []
    for child in range(DECODER_STAGED_FRAME_GROUPS):
        prefix = f"decoder_staged_frame{child}"
        frames.append(plan(
            f"frame{child}",
            (
                _item(f"{prefix}_weights_bf16",
                      DECODER_PHASE_WEIGHT_BYTES_BF16["frame"],
                      "one frame child graph's source F32 weights as BF16"),
                _item(f"{prefix}_parent_input_bf16", history_middle,
                      "one typed middle input"),
                _item(f"{prefix}_workspace_allowance", extra_device_bytes,
                      "compiler workspace and device upload allowance"),
            ),
            (
                *(
                    (
                        _item(f"{prefix}_parent_list_host_bf16",
                              _mul(2 * latent_count, history_middle,
                                   name="staged frame parent list"),
                              "ten history parents retained for source-order scatter"),
                    )
                    if parent_storage == "host" else (
                        _item(f"{prefix}_decoder_model_owners_host_f32", model_owners,
                              "module, checkpoint/model clones retained during staged graph setup"),
                        _item(f"{prefix}_parent_input_host_bf16",
                              _mul(2, history_middle, name="frame parent typed handoff"),
                              "current parent raw array plus typed graph feed"),
                    )
                ),
                _item(f"{prefix}_output_buffer_host_bf16", frame_output,
                      "one graph-owned public BF16 frame output buffer"),
                _item(f"{prefix}_current_output_host_f32", frame_output_f32,
                      "one current frame readback staging owner"),
                *(
                    (_item(f"{prefix}_final_host_f32",
                           _mul(DECODER_FINAL_FRAMES, frame_output_f32,
                                name="staged frame final output"),
                           "seventeen-frame direct-scatter final result owner"),)
                    if parent_storage == "host" else (
                        _item(f"{prefix}_output_readback_host_bf16", frame_output,
                              "public BF16 frame readback owner"),
                        _item(f"{prefix}_normalized_output_owner_host_f32",
                              _mul(output_frames, 3, geometry.height, geometry.width, 4,
                                   name="staged normalized output owner"),
                              "one visible output owner written per frame"),
                        _item(f"{prefix}_normalization_scratch_host_f32",
                              _mul(6, frame_output_f32,
                                   name="staged per-frame normalization scratch"),
                              "reusable scratch for two BF16 RNE boundaries"),
                    )
                ),
                _item(f"{prefix}_typed_io_allowance", extra_host_bytes,
                      "typed host input/output staging owned by the controller"),
            ),
            DECODER_STAGED_FRAME_CHILDREN,
        ))
    return (front, *((grow,) if not fuse_growth else ()), *history, *frames)


def phase_peak_bytes(
    denoiser_bytes: int,
    projection_bytes: int,
    decoder_bytes: int,
    extra_device_bytes: int,
    *,
    grouping: str,
) -> int:
    """Combine known device bytes for one explicit lifecycle grouping."""
    values = (denoiser_bytes, projection_bytes, decoder_bytes, extra_device_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("phase byte values must be nonnegative integers")
    if grouping == "sum":
        return _add(values, "overlapping phase device inventory")
    if grouping == "all_disjoint":
        return _add((max(denoiser_bytes, projection_bytes, decoder_bytes), extra_device_bytes),
                    "disjoint phase device inventory")
    if grouping == "projection_denoiser_vs_decoder":
        return _add((max(denoiser_bytes + projection_bytes, decoder_bytes), extra_device_bytes),
                    "projection/denoiser versus decoder device inventory")
    raise ValueError(
        "phase grouping must be 'sum', 'all_disjoint', or "
        "'projection_denoiser_vs_decoder'"
    )


def estimate_plus_memory(
    geometry: FrameGeometry,
    *,
    mode: str = "turbo",
    disk_history: bool = False,
    prefetched_history: bool = False,
    available_device_bytes: int,
    available_host_bytes: int,
    reserve_device_bytes: int = 0,
    reserve_host_bytes: int = 0,
    resident_weight_layers: int = 1,
    stream_call_count: int = 2,
    remote_pre: bool = False,
    remote_tokens: bool = False,
    post_cross_bridge: bool = False,
    compact_kv: bool = False,
    retained_kv_roi: bool = False,
    split_key: bool = False,
    ffn2_matmul: bool = False,
    post_ffn2_state: bool = False,
    workspace_rows: int = MAX_WORKSPACE_ROWS,
    decoder_phase: bool = True,
    decoder_output_frames: int | None = None,
    decoder_weight_bytes: int = 0,
    decoder_workspace_bytes: int = 0,
    projection_phase: bool = True,
    phase_lifetimes_disjoint: bool = False,
    phase_grouping: str | None = None,
    host_phase_grouping: str | None = None,
    decoder_remote_state: bool = False,
    extra_device_bytes: int = 0,
    extra_host_bytes: int = 0,
    extra_buffers_complete: bool = False,
) -> PlusMemoryPlan:
    """Return a named-buffer inventory without allocating.

    ``available_*`` are caller-supplied free bytes after unrelated processes;
    reserves are explicit untouched headroom, never a hidden percentage. The
    projection inventory covers the steady four-frame LQ request, both causal
    cache pairs, and all eight released projection tensors. The denoiser names
    only row-bounded pre Q/K/V and pooled outputs; full source-order Q/K/V and
    token state are host-staged. Six private pre Result slabs are named in the
    denoiser inventory. ``extra_*`` must cover
    graph/compiler workspaces, packed upload copies, and caller-specific
    source/FP32 diagnostic staging; ``fits`` stays false until the caller
    marks that allowance complete. ``phase_grouping`` may be ``"sum"``
    (the historical conservative inventory), ``"all_disjoint"``, or
    ``"projection_denoiser_vs_decoder"`` for the resident controller's
    proven projection-plus-denoiser overlap followed by decoder teardown.
    ``host_phase_grouping`` defaults to the historical summed host inventory;
    ``"resident"`` selects explicit projection/setup/warm host peaks by
    ``max`` and is rejected for any other controller contract.
    ``compact_kv`` accounts for its proved startup/later-call K/V overlap;
    fixed history capacity, unknown allowances and reserves are unchanged.
    ``retained_kv_roi`` is an opt-in remote-pre contract: the pre GPU roots
    fill the existing CompactHistory destinations directly, so the full
    current K/V host staging owner is absent. It is accepted only with the
    compact resident lifecycle; all fixed histories, allowances, and device
    roots remain charged.
    ``split_key`` is an opt-in resident C16 attention contract. It charges
    one partial-key F32 workspace without changing the existing attention
    output, host inventory, allowances, or reserves.
    ``ffn2_matmul`` is an opt-in POST graph contract. It charges the two
    private BF16 FFN2 MatMul Result slabs at ``workspace_rows``; these outputs
    remain GPU-only and have no host charge.
    ``post_ffn2_state`` replaces those exact two private Result owners with
    two same-shaped BF16 Assign variables. Their full byte charge is retained,
    not credited as an alias; all request/workspace allowances stay unchanged.
    ``post_cross_bridge`` is an opt-in accurate two-branch POST cross bridge.
    It requires the reviewed remote-token 4096-row lifecycle, trades two
    resident PRE/POST banks for a separately named bridge workspace, and does
    not reduce any existing allowance or reserve.
    ``decoder_remote_state=True`` omits the legacy host-F32 state item; the
    two BF16 remote state sets remain in ``decoder_state_io_bf16``.
    ``decoder_output_frames`` selects the authenticated resident streaming
    decoder inventory.  ``None`` preserves the historical fixed-17-frame
    inventory and its two-call validation.
    ``mode`` selects the pinned Turbo/Base layer retention policy. The default
    Turbo policy is byte-identical to the historical fixed 40S inventory.
    ``disk_history`` is a strict resident remote-pre opt-in: full mode-sized
    BF16 K/V history is reported as disk bytes, while host RAM retains only
    one active maximum global pair and all pooled FP32 history lanes.
    ``prefetched_history`` is a stricter disk-history opt-in: the host ledger
    retains at most the current and next maximum global K/V prefixes while the
    next prefix is prefetched after the current file is flushed.
    """
    if not isinstance(geometry, FrameGeometry):
        raise TypeError("geometry must be a FrameGeometry")
    mode_spec = resolve_plus_mode(mode)
    if type(disk_history) is not bool:
        raise ValueError("disk_history must be boolean")
    if type(prefetched_history) is not bool:
        raise ValueError("prefetched_history must be boolean")
    if prefetched_history and not disk_history:
        raise ValueError("prefetched_history requires disk_history")
    available_device_bytes = _budget(available_device_bytes, "available_device_bytes")
    available_host_bytes = _budget(available_host_bytes, "available_host_bytes")
    reserve_device_bytes = _budget(reserve_device_bytes, "reserve_device_bytes")
    reserve_host_bytes = _budget(reserve_host_bytes, "reserve_host_bytes")
    if reserve_device_bytes > available_device_bytes or reserve_host_bytes > available_host_bytes:
        raise ValueError("memory reserve cannot exceed corresponding availability")
    resident_weight_layers = _int(resident_weight_layers, "resident_weight_layers", positive=True)
    if resident_weight_layers > 30:
        raise ValueError("resident_weight_layers cannot exceed the 30 Plus blocks")
    stream_call_count = _int(stream_call_count, "stream_call_count", positive=True)
    if not 2 <= stream_call_count <= 4:
        raise ValueError("stream_call_count must match the bounded two-to-four-call input ABI")
    if type(remote_pre) is not bool:
        raise ValueError("remote_pre must be boolean")
    if type(remote_tokens) is not bool:
        raise ValueError("remote_tokens must be boolean")
    if type(post_cross_bridge) is not bool:
        raise ValueError("post_cross_bridge must be boolean")
    if type(compact_kv) is not bool:
        raise ValueError("compact_kv must be boolean")
    if type(retained_kv_roi) is not bool:
        raise ValueError("retained_kv_roi must be boolean")
    if type(split_key) is not bool:
        raise ValueError("split_key must be boolean")
    if type(ffn2_matmul) is not bool:
        raise ValueError("ffn2_matmul must be boolean")
    if type(post_ffn2_state) is not bool:
        raise ValueError("post_ffn2_state must be boolean")
    if post_ffn2_state and (not ffn2_matmul or not post_cross_bridge):
        raise ValueError("post_ffn2_state requires ffn2_matmul and post_cross_bridge")
    workspace_rows = _int(workspace_rows, "workspace_rows", positive=True)
    # Keep the historical default at 4096 while allowing an explicit 8192
    # slab; larger rows have not been reviewed for the bounded graph ABI.
    if workspace_rows > 2 * MAX_WORKSPACE_ROWS or workspace_rows % BLOCK_TOKENS:
        raise ValueError("workspace_rows must be whole 128-token slabs no larger than 8192")
    if not isinstance(decoder_phase, bool):
        raise ValueError("decoder_phase must be boolean")
    if decoder_output_frames is not None and not decoder_phase:
        raise ValueError("decoder_output_frames requires decoder_phase=True")
    stream_plan = None
    if decoder_output_frames is not None:
        stream_plan = _stream_decoder_plan(
            geometry, decoder_output_frames, stream_call_count
        )
    if decoder_phase and decoder_output_frames is None and stream_call_count != DECODER_CHUNKS:
        raise ValueError("The current unphased decoder inventory requires two stream calls")
    decoder_weight_bytes = _budget(decoder_weight_bytes, "decoder_weight_bytes")
    decoder_workspace_bytes = _budget(decoder_workspace_bytes, "decoder_workspace_bytes")
    if not decoder_phase and (decoder_weight_bytes or decoder_workspace_bytes):
        raise ValueError("decoder byte allowances require decoder_phase=True")
    if not isinstance(projection_phase, bool):
        raise ValueError("projection_phase must be boolean")
    if not isinstance(phase_lifetimes_disjoint, bool):
        raise ValueError("phase_lifetimes_disjoint must be boolean")
    if phase_grouping is not None and not isinstance(phase_grouping, str):
        raise ValueError("phase_grouping must be a string or None")
    grouping = "all_disjoint" if phase_lifetimes_disjoint else "sum"
    if phase_grouping is not None:
        if phase_lifetimes_disjoint and phase_grouping != "all_disjoint":
            raise ValueError("phase_lifetimes_disjoint conflicts with phase_grouping")
        grouping = phase_grouping
    if grouping not in ("sum", "all_disjoint", "projection_denoiser_vs_decoder"):
        raise ValueError(
            "phase_grouping must be 'sum', 'all_disjoint', or "
            "'projection_denoiser_vs_decoder'"
        )
    if host_phase_grouping is not None and not isinstance(host_phase_grouping, str):
        raise ValueError("host_phase_grouping must be a string or None")
    if host_phase_grouping not in (None, "sum", "resident"):
        raise ValueError("host_phase_grouping must be 'sum' or 'resident'")
    if compact_kv and (not remote_pre or host_phase_grouping != "resident"):
        raise ValueError("compact_kv requires the resident remote-pre lifecycle")
    if retained_kv_roi and (
        not compact_kv or not remote_pre or host_phase_grouping != "resident"
    ):
        raise ValueError(
            "retained_kv_roi requires compact_kv with the resident remote-pre lifecycle"
        )
    if disk_history and (
        host_phase_grouping != "resident"
        or remote_pre is not True
        or compact_kv is not True
        or retained_kv_roi is not True
    ):
        raise ValueError(
            "disk_history requires the resident remote-pre compact retained-ROI lifecycle"
        )
    if host_phase_grouping == "resident" and (
        remote_pre is not True
        or resident_weight_layers != 30
        or projection_phase is not True
        or decoder_phase is not False
        or phase_lifetimes_disjoint is not True
        or phase_grouping != "all_disjoint"
    ):
        raise ValueError(
            "resident host phase grouping requires remote_pre, 30 resident layers, "
            "projection-only execution, and explicitly disjoint GPU phases"
        )
    if split_key and (remote_pre is not True or host_phase_grouping != "resident"):
        raise ValueError(
            "split_key requires the resident remote-pre C16 lifecycle"
        )
    if not isinstance(decoder_remote_state, bool):
        raise ValueError("decoder_remote_state must be boolean")
    if decoder_remote_state and not decoder_phase:
        raise ValueError("decoder_remote_state requires decoder_phase=True")
    if not isinstance(extra_buffers_complete, bool):
        raise ValueError("extra_buffers_complete must be boolean")
    extra_device_bytes = _budget(extra_device_bytes, "extra_device_bytes")
    extra_host_bytes = _budget(extra_host_bytes, "extra_host_bytes")
    if host_phase_grouping == "resident" and (
        extra_device_bytes < (4 << 30) or extra_host_bytes < (4 << 30)
    ):
        raise ValueError(
            "resident host phase grouping requires at least 4 GiB of each "
            "explicit device/host allowance"
        )
    if remote_tokens and (host_phase_grouping != "resident" or workspace_rows != 4096
                          or max(geometry.token_hw) > 240
                          or math.prod(geometry.token_hw) > 136 * 240):
        raise ValueError("remote_tokens requires the reviewed resident f4/4096-row lifecycle")
    if post_cross_bridge and (
        not remote_tokens or not remote_pre or workspace_rows != MAX_WORKSPACE_ROWS
        or resident_weight_layers != 30
    ):
        raise ValueError(
            "post_cross_bridge requires remote_tokens, remote_pre, 4096 rows, "
            "and all 30 resident weight layers"
        )

    s = _int(geometry.spatial_windows, "spatial_windows", positive=True)
    if geometry.spatial_tokens % 64 or geometry.spatial_tokens // 64 != s:
        raise ValueError("geometry must contain complete 8x8 token windows")
    total_weights, global_weights, layer_weights = _weight_elements()
    active_weights = global_weights + resident_weight_layers * layer_weights
    weight_staging = 0
    if remote_tokens:
        from plus_weights import EXTRA_LAYER_SHAPES
        cross_weights = sum(math.prod(shape) for shape in EXTRA_LAYER_SHAPES.values())
        pre_post_weights = layer_weights - cross_weights
        # The default has 26 permanent PRE/POST banks plus one active bank;
        # the accurate bridge uses 24 permanent banks plus one refill bank.
        # Count the setup-only cross input bank even alongside the token root.
        resident_banks = (POST_CROSS_BRIDGE_RESIDENT_BANKS
                          if post_cross_bridge else 27)
        active_weights = global_weights + resident_banks * pre_post_weights + cross_weights
        weight_staging = layer_weights * 4 + pre_post_weights * 2
    projection_weights = _projection_elements()
    storage = 2
    block_values = _mul(BLOCK_TOKENS, CHANNELS, name="Plus block")
    pool_values = _mul(HEADS, HEAD_DIM, name="pooled head")
    current_blocks = _mul(CURRENT_TEMPORAL_WINDOWS, s, name="current blocks")
    history_group_count = _add(
        [mode_spec.retain_groups(layer) for layer in range(GLOBAL_LAYERS + LOCAL_LAYERS)],
        "mode rolling history groups",
    )
    history_blocks = _mul(history_group_count, s, name="host rolling history")
    max_cache_blocks = _mul((GLOBAL_RETENTION + CURRENT_TEMPORAL_WINDOWS), s,
                            name="device retained cache")
    tokens = _mul(4, geometry.spatial_tokens, name="current tokens")
    if remote_pre and (workspace_rows != 4096 or tokens < workspace_rows):
        raise ValueError("remote_pre requires the4096-row/C16 producer layout")
    qkv = _mul(3, current_blocks, block_values, storage, name="current QKV host")
    qkv_device = _mul(3, workspace_rows, CHANNELS, storage,
                      name="bounded pre QKV outputs")
    token_state = _mul(2, tokens, CHANNELS, storage, name="full token state")
    cache = _mul(2, max_cache_blocks, block_values, storage, name="current K/V cache")
    pooled = _mul(current_blocks + max_cache_blocks, pool_values, 4, name="pooled mask workspace")
    rolling_kv_full = _mul(
        2, history_blocks, block_values, storage, name="host rolling K/V"
    )
    rolling_kv = rolling_kv_full
    history_disk_bytes = 0
    if disk_history:
        active_global_groups = max(
            mode_spec.retain_groups(layer)
            for layer in range(GLOBAL_LAYERS + LOCAL_LAYERS)
        )
        active_global_blocks = _mul(
            2, active_global_groups, s, block_values, storage,
            name="active global K/V pair",
        )
        rolling_kv = active_global_blocks
        if prefetched_history:
            rolling_kv = _mul(
                2, active_global_blocks,
                name="current plus prefetched global K/V pairs",
            )
        history_disk_bytes = rolling_kv_full
    rolling_pooled = _mul(history_blocks, pool_values, 4, name="host rolling pooled")
    mask = _mul(HEADS, current_blocks, max_cache_blocks, 1, name="block mask")
    ids = _mul(HEADS, ATTENTION_QUERY_WINDOWS, 1, max_cache_blocks, 4,
               name="packed query block ids")
    mask_ids = _mul(HEADS, ATTENTION_QUERY_WINDOWS, 1, max_cache_blocks, 4,
                    name="packed block ids host")
    pooled_outputs = _mul(
        2, HEADS, workspace_rows // BLOCK_TOKENS, HEAD_DIM, 4,
        name="bounded pre pooled outputs",
    )
    h, w = geometry.working_hw
    video_bytes = _mul(3, 4, h, w, storage, name="LQ projection video")

    device = (
        _item("dit_active_weights_bf16", _mul(active_weights, storage),
              (f"shared globals, {resident_banks} PRE/POST banks and one setup cross input bank"
               if remote_tokens else
              "shared 15 tensors plus the active layer; set resident_weight_layers=30 for all 825 on device")),
        _item("current_qkv_bf16", qkv_device,
              "one bounded pre request's Q/K/V output slabs; full Q/K/V are host-staged"),
        _item("current_kv_cache_bf16", cache, "worst global cache: three history groups plus two current groups"),
        _item("resident_attention_output_bf16", _mul(
              HEADS, ATTENTION_QUERY_WINDOWS, BLOCK_TOKENS, HEAD_DIM, storage,
              name="resident attention output"),
              "one persistent output of the fixed query_windows=16 resident attention graph"),
        _item("resident_cross_kv_bf16", _mul(2, 30, 512, CHANNELS, storage,
              name="resident cross K/V"),
              "all 30 cross-K/V pairs retained by run_plus_resident until denoiser teardown"),
        _item("cross_kv_graph_output_bf16", _mul(2, 512, CHANNELS, storage,
              name="cross graph output K/V"),
              "one transient cross graph output pair while each layer is uploaded"),
        _item("packed_block_ids_i32", ids,
              "one fixed query_windows=16 XMX block-list input at full-cache capacity"),
        _item("pre_pooled_outputs_f32", pooled_outputs,
              "bounded pre q/k pooled output pair, [12, rows/128, 128] FP32"),
        _item("bounded_pre_workspace_bf16", _mul(2, workspace_rows, CHANNELS, storage, name="pre workspace"),
              "one row slab input/output"),
        _item("bounded_post_ffn_workspace_bf16", _mul(2, workspace_rows, FFN_HIDDEN, storage, name="post FFN workspace"),
              "one row slab through source 8960-wide FFN"),
        _item("bounded_post_token_workspace_bf16", _mul(2, workspace_rows, CHANNELS, storage, name="post token workspace"),
              "one row slab around post residual/cross paths"),
        _item("pre_private_storage_results_bf16", _mul(6, workspace_rows, CHANNELS, storage,
              name="pre private Result slabs"),
              "six source-boundary pre Results: product, q/k Linear, norm1, q/k weighted RMS"),
        _item("extra_device_buffers_allowance", extra_device_bytes,
              "caller allowance for graph workspaces, packed uploads, and phase-specific device copies"),
    )
    if post_cross_bridge:
        device += (_item(
            "post_cross_bridge_workspace",
            POST_CROSS_BRIDGE_WORKSPACE_BYTES,
            "two accurate BF16 cross-bridge branch workspaces; no alias credit",
        ),)
    if ffn2_matmul:
        device += (_item(
            "ffn2_private_states_bf16" if post_ffn2_state else "ffn2_private_results_bf16",
            _mul(2, workspace_rows, CHANNELS, storage,
                 name="FFN2 private Result slabs"),
            ("two owning POST FFN2 BF16 Assign states replace the two removed Result roots; no memory credit"
             if post_ffn2_state else
             "two POST FFN2 MatMul Results retained on GPU; no host owner"),
        ),)
    if remote_pre:
        # Each 32-window producer submission owns two C16 Q outputs, including
        # the padded final pair. BF16/F16 views alias the same allocations.
        query_slots = 2 * ((current_blocks + 31) // 32)
        device += (
            _item("resident_q_slots_bf16", _mul(
                  query_slots, HEADS, ATTENTION_QUERY_WINDOWS, BLOCK_TOKENS, HEAD_DIM, storage,
                  name="resident Q slots"),
                  "all current Q windows retained in even pairs of C16 BF16 GPU slots"),
            _item("remote_pre_tail_kv_scratch_bf16", _mul(
                  2, 32, BLOCK_TOKENS, CHANNELS, storage, name="remote pre tail K/V scratch"),
                  "one fixed32-window K/V pair for a partially filled final producer slab"),
        )
    if remote_tokens:
        device += (_item("resident_token_root_bf16", _mul(
            (current_blocks + 31) // 32, 4096, CHANNELS, storage,
            name="padded remote tokens"),
            "one window-major x root; post writes its existing distinct slab"),)
    if split_key:
        device += (_item(
            "attention_split_key_partial_f32",
            _mul(4, HEADS, ATTENTION_QUERY_WINDOWS, BLOCK_TOKENS, 130, 4,
                 name="attention split-key partial F32 workspace"),
            "four partial-key F32 slabs; separate from the existing attention output",
        ),)
    host = (
        _item("dit_all_weights_bf16", _mul(total_weights, storage, name="all Plus weights"),
              "all 825 source shapes retained/mapped on host"),
        _item("full_token_state_host_bf16", token_state, "host-side current x/context copy"),
        _item(
            "current_qkv_host_bf16",
            0 if retained_kv_roi else (qkv // 3 * 2 if remote_pre else qkv),
            ("direct retained GPU-root K/V fill into CompactHistory-owned destinations; "
             "no full current K/V host staging"
             if retained_kv_roi else
             "fresh K/V history staging only; Q stays GPU-only" if remote_pre
             else "host-side current Q/K/V staging"),
        ),
        _item("attention_query_input_host_bf16", 0 if remote_pre else _mul(
              HEADS, ATTENTION_QUERY_WINDOWS, BLOCK_TOKENS, HEAD_DIM, storage,
              name="attention query input staging"),
              "one host-shared [12,16,128,128] BF16 query staging per attention request"),
        _item(
            "rolling_kv_host_bf16",
            rolling_kv,
            (
                ("two active maximum global K/V pairs (current plus prefetched next "
                 "prefix); full mode history is disk-backed"
                 if prefetched_history else
                 "one active maximum global K/V pair; full mode history is disk-backed")
                if disk_history else
                ("5 global layers*3 groups + 25 local layers*1 group" if mode == "turbo" else
                 f"mode {mode_spec.name}: {history_group_count} retained K/V groups")
            ),
        ),
        _item("rolling_pooled_host_f32", rolling_pooled, "one retained pooled K lane per history block"),
        _item("pooled_mask_workspace_f32", pooled, "current q plus worst merged k pooled arrays"),
        _item("block_mask_host_bool", mask, "source-order block mask before XMX packing"),
        _item("packed_block_ids_host_i32", mask_ids,
              "one fixed query_windows=16 packed ID request; the full block mask remains host-resident"),
        _item("lq_projection_weights_host_f32", _mul(projection_weights, 4,
              name="host LQ projection weights"),
              "eight source LQ tensors loaded as F32 before BF16 graph constants"),
        _item("lq_projection_video_host_bf16", video_bytes,
              "bounded four-frame BF16 LQ input; F32 diagnostic video staging is caller extra"),
        _item("lq_projection_conditions_host_bf16", _mul(
              stream_call_count, 4, geometry.spatial_tokens, CHANNELS, storage,
              name="host projection conditions"),
              "all stream calls' four-latent BF16 conditions retained across projection-first handoff"),
        _item("extra_host_buffers_allowance", extra_host_bytes,
              "caller allowance for source/FP32 staging, readback, and packed host copies"),
    )

    projection = (
        _item("lq_projection_weights_bf16", _mul(projection_weights, storage,
              name="LQ projection weights"),
              "all eight released Plus LQ projection tensors; may be released before DiT"),
        _item("lq_projection_causal_states_bf16", _mul(
              2, 2, PROJECTION_INPUT_CHANNELS + PROJECTION_CACHE_CHANNELS,
              geometry.spatial_tokens, storage, name="LQ causal states"),
              "previous and next cache pairs, two temporal states in each pair"),
        _item("lq_projection_video_device_bf16", video_bytes,
              "steady four-frame BF16 projection input on device"),
    ) if projection_phase else ()

    decoder = ()
    decoder_host = ()
    if decoder_phase:
        packed = _mul(784, h // 8, w // 8, storage, name="decoder packed input")
        state_values = (3 * 512 * (h // 8) * (w // 8)
                        + 3 * 256 * (h // 4) * (w // 4)
                        + 3 * 128 * (h // 2) * (w // 2))
        states = _mul(2, state_values, storage, name="decoder states")
        frames = _mul(4, 3, h, w, storage, name="decoder output")
        private_residual_values = _add(
            [_mul(groups, channels, h // divisor, w // divisor,
                  name="decoder private residual Result")
             for groups, channels, divisor in DECODER_PRIVATE_RESIDUAL_GROUPS],
            "decoder private residual Results",
        )
        private_residual_bytes = _mul(private_residual_values, storage,
                                      name="decoder private residual Results")
        decoder = (
            _item("decoder_packed_input_bf16", packed, "one [1,784,H/8,W/8] packed latent"),
            _item("decoder_state_io_bf16", states, "nine explicit state inputs/outputs, both sides live at handoff"),
            _item("decoder_output_bf16", frames, "one four-frame RGB output"),
            _item("decoder_private_residual_results_bf16", private_residual_bytes,
                  "twelve source residual-linear Result buffers: 3x512 at H/8, 3x256 at H/4, 6x128 at H/2"),
            _item("decoder_weights_bf16", decoder_weight_bytes, "caller-supplied; source graph does not prove this peak"),
            _item("decoder_compiler_workspace_bytes", decoder_workspace_bytes,
                  "caller-supplied; no OpenVINO allocator peak is inferred"),
        )
        if stream_plan is None:
            decoder_host_items = [
                _item("decoder_module_host_f32", DECODER_PARAMETER_BYTES_F32,
                      "authenticated 66-tensor TCDecoder module remains F32 during graph build and decode"),
                _item("decoder_source_video_host_f32", _mul(
                      DECODER_SOURCE_FRAMES, 3, h, w, 4, name="decoder source video"),
                      "33-frame common F32 source retained by run_plus_resident.decode"),
                _item("decoder_denoised_chunks_host_bf16", _mul(
                      DECODER_CHUNKS, DECODER_CHUNK_FRAMES, 4, h // 8, w // 8, storage,
                      name="decoder denoised chunks"),
                      "two raw BF16 denoised chunks (16x4 latent ABI) retained by the caller during decode"),
                _item("decoder_current_denoised_host_f32", _mul(
                      DECODER_CHUNK_FRAMES, 4, h // 8, w // 8, 4,
                      name="decoder current denoised"),
                      "one 16x4 latent chunk decoded to F32 before pack_decoder_inputs"),
                _item("decoder_packed_input_host_f32", _mul(
                      4, 784, h // 8, w // 8, 4, name="decoder packed host input"),
                      "one four-latent F32 packed batch retained across inner requests"),
                _item("decoder_frame_accum_host_f32", _mul(
                      DECODER_OUTPUT_FRAMES, 3, h, w, 4, name="decoder frame accumulation"),
                      "29 F32 frames retained before final 17-frame trim"),
                _item("decoder_final_host_f32", _mul(
                      DECODER_FINAL_FRAMES, 3, h, w, 4, name="decoder final frames"),
                      "trimmed raw F32 output held while normalized output is formed"),
                _item("decoder_normalized_output_host_f32", _mul(
                      DECODER_FINAL_FRAMES, 3, h, w, 4, name="decoder normalized output"),
                      "final F32 output retained for the compressed result archive"),
            ]
            if not decoder_remote_state:
                decoder_host_items.insert(
                    4,
                    _item("decoder_state_io_host_f32", _mul(
                          2, state_values, 4, name="decoder host states"),
                          "legacy host-F32 state formula; omit when controller binds remote BF16 state"),
                )
        else:
            # The resident stream controller keeps one source/video owner and
            # writes each four-frame result into one reusable host-bound lane.
            # No frame list, 29-frame accumulator, or F32 packed latent copy is
            # part of this mode's ownership contract.
            projection_frames = stream_plan.projection_frames
            output_frames = stream_plan.output_frames
            call_count = stream_plan.call_count
            source_video = _mul(3, projection_frames, h, w, 4,
                                name="stream decoder source video")
            denoised_chunks = _mul(
                call_count, DECODER_CHUNK_FRAMES, 4, h // 8, w // 8, storage,
                name="stream decoder denoised chunks",
            )
            state_bf16 = _mul(2, state_values, storage,
                               name="stream decoder host states")
            bound_frame = _mul(4, 3, h, w, storage,
                               name="stream decoder bound frame")
            result_f32 = _mul(4, 3, h, w, 4,
                              name="stream decoder F32 result")
            packed_bf16 = _mul(784, h // 8, w // 8, storage,
                               name="stream decoder packed input")
            selected_conditioning = _mul(4, 3, h, w, storage,
                                         name="stream decoder conditioning")
            normalized = _mul(output_frames, 3, geometry.height,
                              geometry.width, 4,
                              name="stream decoder normalized output")
            # 32 bytes per four-frame pixel/channel is a deliberately
            # conservative allowance for normalization scratch, including
            # transient views of one decoder result chunk.
            normalization_scratch = _mul(
                32, 4, 3, geometry.height, geometry.width,
                name="stream decoder normalization scratch",
            )
            decoder_host_items = [
                _item("decoder_stream_module_host_f32", DECODER_PARAMETER_BYTES_F32,
                      "authenticated decoder module owner during graph build"),
                _item("decoder_stream_state_owner_host_f32", DECODER_PARAMETER_BYTES_F32,
                      "transient F32 state-map owner during graph build"),
                _item("decoder_stream_source_video_host_f32", source_video,
                      "source projection frames retained for the authenticated stream"),
                _item("decoder_stream_denoised_chunks_host_bf16", denoised_chunks,
                      "one raw BF16 16x4 chunk per fixed-four-latent stream call"),
                _item("decoder_stream_state_io_host_bf16", state_bf16,
                      "two BF16 ping-pong state sets carried between decoder calls"),
                _item("decoder_stream_normalized_output_host_f32", normalized,
                      "one visible source-frame output owner; no frame accumulator"),
                _item("decoder_stream_bound_frame_host_bf16", bound_frame,
                      "one host-bound BF16 four-frame decoder result"),
                _item("decoder_stream_raw_readback_host_bf16", bound_frame,
                      "one BF16 readback copy for the bound decoder result"),
                _item("decoder_stream_result_host_f32", result_f32,
                      "one transient F32 result raster for frame normalization"),
                _item("decoder_stream_packed_input_host_bf16", packed_bf16,
                      "one raw BF16 packed latent input"),
                _item("decoder_stream_packed_input_typed_bf16", packed_bf16,
                      "one typed host BF16 feed view/copy"),
                _item("decoder_stream_selected_conditioning_host_bf16",
                      selected_conditioning,
                      "one selected four-frame BF16 conditioning staging"),
                _item("decoder_stream_normalization_scratch_host_bytes",
                      normalization_scratch,
                      "conservative 32 bytes per four-frame visible pixel/channel for normalization"),
            ]
        decoder_host = tuple(decoder_host_items)
    denoiser_required = _add([item.bytes for item in device], "denoiser device inventory")
    denoiser_known = denoiser_required - extra_device_bytes
    projection_known = _add([item.bytes for item in projection], "projection device inventory")
    decoder_required = _add([item.bytes for item in decoder], "decoder device inventory")
    required_device = phase_peak_bytes(
        denoiser_known,
        projection_known if projection_phase else 0,
        decoder_required if decoder_phase else 0,
        extra_device_bytes,
        grouping=grouping,
    )
    legacy_required_host = _add(
        [item.bytes for item in (*host, *decoder_host)],
        "host inventory",
    )
    host_grouping = "resident" if host_phase_grouping == "resident" else "sum"
    host_phase_peaks: tuple[MemoryItem, ...] = ()
    if host_grouping == "resident":
        # The explicit remote-pre lifecycle has three host peaks:
        # source/LQ projection, checkpoint/graph setup, then warm denoising.
        # Keep the four-GiB unknown allowance in every phase; it is not reduced
        # merely because the named buffers are disjoint.
        aux_inputs = _add((
            _mul(512, 4096, 4, name="resident prompt input"),
            _mul(stream_call_count, 16, 4, h // 8, w // 8, 4,
                 name="resident noise inputs"),
            _mul(stream_call_count, 2, 4, geometry.spatial_tokens, 64, 4,
                 name="resident RoPE inputs"),
        ), "resident auxiliary input arrays")
        all_conditions = _mul(
            stream_call_count, 4, geometry.spatial_tokens, CHANNELS, storage,
            name="resident all projection conditions",
        )
        projection_source = _mul(
            3, 1 + stream_call_count * 16, h, w, 4,
            name="resident projection source",
        )
        projection_peak = _add((
            projection_source,
            aux_inputs,
            _mul(projection_weights, 4, name="resident LQ F32 weights"),
            all_conditions,
            video_bytes,
            extra_host_bytes,
        ), "resident projection host peak")

        setup_peak = _add((
            _mul(total_weights, 4, name="resident DiT F32 checkpoint mapping"),
            _mul(layer_weights, 10,
                 name="resident two F32 layers plus BF16 conversion"),
            _mul(global_weights, 4, name="resident edge/text F32 temporaries"),
            aux_inputs,
            all_conditions,
            extra_host_bytes,
        ), "resident denoiser setup host peak")

        warm_condition_credit = int(compact_kv and mode_spec.name == "turbo")
        warm_conditions = _mul(
            stream_call_count - 1 - warm_condition_credit,
            4, geometry.spatial_tokens, CHANNELS, storage,
            name="resident warm projection conditions",
        )
        # Compact startup peaks at 40S retained + 1S current local K/V;
        # the last global layer precedes five not-yet-allocated local owners.
        # Later calls can have 40S + 2S current K/V, but one fewer condition.
        # One S K/V pair equals one four-latent condition, so this bound keeps
        # full 2S staging and charges one fewer condition in the compact path.
        denoised_chunks = _mul(
            stream_call_count, 16, 4, h // 8, w // 8, storage,
            name="resident denoised chunks",
        )
        warm_peak = _add((
            warm_conditions,
            rolling_kv,
            rolling_pooled,
            token_state,
            0 if retained_kv_roi else qkv // 3 * 2,
            aux_inputs,
            denoised_chunks,
            pooled,
            mask,
            mask_ids,
            weight_staging,
            extra_host_bytes,
        ), "resident denoiser warm host peak")
        host_phase_peaks = (
            _item("host_projection_phase_peak", projection_peak,
                  "source video, LQ weights, all conditions, auxiliary inputs, and unknown allowance"),
            _item("host_denoiser_setup_phase_peak", setup_peak,
                  "F32 checkpoint mapping, two F32 layer copies, BF16 conversion, edge/text temporaries, and all conditions"),
            _item("host_denoiser_warm_phase_peak", warm_peak,
                  (("compact startup/later-call overlap, 40S histories, direct retained K/V owner fill, "
                    "inputs, chunks, masks, and allowance") if retained_kv_roi else
                   "compact startup/later-call overlap, 40S histories, staging, inputs, chunks, masks, and allowance")
                  if compact_kv else
                  "C-1 conditions, compact 40S histories, token/KV staging, auxiliary inputs, chunks, masks, and allowance"),
        )
        if remote_tokens:
            host += (_item("remote_token_weight_staging_host", weight_staging,
                           "one F32 source layer and a complete BF16 PRE/POST staging bank"),)
        required_host = max(item.bytes for item in host_phase_peaks)
    else:
        required_host = legacy_required_host
    assumptions = (
        "BF16 tensors use two bytes; pooled mask lanes use FP32 four bytes",
        "all 825 DiT headers are host-resident; device residency is shared globals plus active layers",
        "host K/V retains 5 global layers at 3 groups and 25 local layers at 1 group",
        "the LQ projection phase names all 8 source tensors and steady previous/next causal cache pairs",
        "bounded pre Q/K/V and pooled outputs use workspace_rows; full source-order Q/K/V and x/context remain host-staged",
        "six private pre Result slabs are counted at workspace_rows; source/FP32 diagnostic video is extra",
        "the fixed query block-list is one [12,16,1,5S] request; the full host block mask is retained for source-order packing",
        "the fixed attention Q input staging is one host-shared [12,16,128,128] BF16 buffer; any plugin copy remains in the explicit device allowance",
        "one fixed query_windows=16 attention output is named; Q/K/V and packed IDs are caller-owned, while graph workspaces remain in extra device allowance",
        "all 30 resident cross-K/V pairs plus one transient cross-graph output pair remain live during denoising",
        ("decoder host items retain the F32 module, 33-frame source, raw chunks, and output staging; remote state is device-only when requested"
         if stream_plan is None else
         f"stream decoder retains {stream_plan.projection_frames} projection frames, "
         f"{stream_plan.call_count} four-latent chunks, two BF16 state sets, and "
         "one visible FP32 output owner"),
        ("resident host phases use max(projection, denoiser setup, denoiser warm); "
         "each phase retains the explicit unknown host allowance"
         if host_grouping == "resident" else
         "host inventories are summed conservatively; the resident terminal call clears histories per layer before decoder handoff"),
        (f"selected device phase grouping: {grouping}; host phases use an explicit maximum"
         if host_grouping == "resident" else
         f"selected device phase grouping: {grouping}; host inventories remain conservatively summed"),
        "extra allowances cover OpenVINO graph workspaces, packed uploads, and unmodelled host/device staging",
        "the inventory is not an allocator peak proof, even when fits is true",
    )
    if stream_plan is not None:
        assumptions += (
            "stream decoder mode replaces the legacy frame accumulator/final/F32-pack items",
            "stream decoder normalization scratch is a conservative 32 bytes per visible pixel/channel",
        )
    if host_grouping == "resident":
        assumptions += (
            "resident host admission requires source pixels deleted before DiT setup and CPU checkpoint mapping closed before histories",
            ("resident warm peak assumes each consumed condition chunk is dropped, post output reuses x, and CompactHistory owns one 40S suffix per layer"
             if mode == "turbo" and not disk_history else
             "resident warm peak drops consumed conditions and reuses x; mapped K/V pages are synchronously flushed/evicted after each active layer"),
        )
    if remote_tokens:
        assumptions += (
            "one padded GPU x root replaces repeated host window gathers/readbacks; existing distinct post output slab is reused",
            ("24 permanent PRE/POST banks plus one refill bank for the accurate cross bridge; cross inputs are retired after cached outputs"
             if post_cross_bridge else
             "26 permanent PRE/POST weight banks plus one active bank for layers26..29; cross inputs are retired after cached outputs"),
            "warm host peak additionally retains one complete F32 layer and the whole BF16 PRE/POST staging bank before refill",
        )
    if post_cross_bridge:
        assumptions += (
            "accurate POST cross bridge trades two resident PRE/POST banks for a named two-branch workspace; existing allowances and reserves are unchanged",
        )
    if retained_kv_roi:
        assumptions += (
            "retained K/V ROIs fill the existing CompactHistory owners directly; the full current K/V host staging item is zero",
            "retained K/V mode leaves rolling history, device roots, readback allowance, and reserves unchanged",
        )
    if mode_spec.name != "turbo":
        assumptions += (
            f"selected Plus mode {mode_spec.name} derives {history_group_count} retained K/V groups from its pinned layer policy",
        )
    if disk_history:
        assumptions += (
            f"disk history stores {history_disk_bytes} BF16 K/V bytes; RAM retains "
            f"{'two' if prefetched_history else 'one'} active global K/V pair(s) and all pooled FP32 lanes",
            "Base warm host accounting keeps the conservative C-1 condition count; Turbo-only compact startup credit is not reused",
        )
    if prefetched_history:
        assumptions += (
            "prefetch history retains at most current plus next disk-backed global K/V prefixes; "
            "the next prefix is prefetched only after the current flush",
        )
    if split_key:
        assumptions += (
            "split-key mode charges one partial-key F32 workspace; existing attention output, host items, allowances, and reserves are unchanged",
        )
    return PlusMemoryPlan(
        geometry=geometry,
        device_items=device,
        host_items=host,
        projection_items=projection,
        decoder_items=decoder,
        decoder_host_items=decoder_host,
        available_device_bytes=available_device_bytes,
        available_host_bytes=available_host_bytes,
        reserve_device_bytes=reserve_device_bytes,
        reserve_host_bytes=reserve_host_bytes,
        required_device_bytes=required_device,
        required_host_bytes=required_host,
        projection_phase=projection_phase,
        decoder_phase=decoder_phase,
        decoder_remote_state=decoder_remote_state,
        phase_grouping=grouping,
        phase_lifetimes_disjoint=phase_lifetimes_disjoint,
        extra_buffers_complete=extra_buffers_complete,
        assumptions=assumptions,
        host_phase_grouping=host_grouping,
        host_phase_peaks=host_phase_peaks,
        remote_tokens=remote_tokens,
        post_cross_bridge=post_cross_bridge,
        post_ffn2_state=post_ffn2_state,
        mode=mode_spec.name,
        disk_history=disk_history,
        prefetched_history=prefetched_history,
        history_disk_bytes=history_disk_bytes,
    )


def admit_plus_memory(*args, **kwargs) -> PlusMemoryPlan:
    """Build a plan and reject incomplete or over-budget admission before work."""
    plan = estimate_plus_memory(*args, **kwargs)
    if not plan.fits:
        if not plan.extra_buffers_complete:
            raise MemoryError(
                "Plus memory admission failed: explicit extra device/host buffer "
                "allowances are missing or not marked complete"
            )
        raise MemoryError(
            f"Plus memory admission failed: device {plan.required_device_bytes} > "
            f"{plan.available_device_bytes - plan.reserve_device_bytes} or host "
            f"{plan.required_host_bytes} > {plan.available_host_bytes - plan.reserve_host_bytes}"
        )
    return plan


__all__ = [
    "MemoryItem", "PlusMemoryPlan", "DecoderPhaseMemory", "StagedDecoderMemory",
    "DECODER_PHASE_ALIASES", "DECODER_PHASE_WEIGHT_BYTES_BF16",
    "DECODER_GROW_WEIGHT_BYTES_BF16", "DECODER_HISTORY_MEMBLOCK_WEIGHT_BYTES_BF16",
    "DECODER_STAGED_LATENTS", "DECODER_STAGED_GROW_CHILDREN",
    "DECODER_STAGED_FRAME_GROUPS", "DECODER_STAGED_FRAME_CHILDREN",
    "DECODER_STAGED_DISK_RESERVE_BYTES", "DECODER_STAGED_HOST_RESERVE_BYTES",
    "DECODER_STAGED_DEVICE_RESERVE_BYTES",
    "POST_CROSS_BRIDGE_RESIDENT_BANKS", "POST_CROSS_BRIDGE_WORKSPACE_BYTES",
    "ATTENTION_QUERY_WINDOWS",
    "decoder_phase_memory",
    "decoder_sequential_memory", "staged_decoder_memory",
    "estimate_plus_memory", "admit_plus_memory",
    "phase_peak_bytes",
]
