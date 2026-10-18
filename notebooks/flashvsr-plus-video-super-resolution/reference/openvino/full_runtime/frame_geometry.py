"""Visible frame geometry and the source model's 128-pixel window alignment."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class FrameGeometry:
    height:int
    width:int

    def __post_init__(self):
        if any(type(v)!=int or v<2 or v%2 for v in (self.height,self.width)):
            raise ValueError('Visible dimensions must be positive even integers')
        h,w=self.working_hw
        if max(h,w)>3840 or h*w>2176*3840:
            raise ValueError('Frame exceeds reviewed padded-UHD memory geometry')

    @property
    def working_hw(self):return tuple((v+127)//128*128 for v in (self.height,self.width))

    @property
    def token_hw(self):return tuple(v//16 for v in self.working_hw)

    @property
    def latent_hw(self):return tuple(v//8 for v in self.working_hw)

    @property
    def spatial_tokens(self):return int(np.prod(self.token_hw))

    @property
    def spatial_windows(self):return self.spatial_tokens//64

    def key(self,frames):
        if type(frames)!=int or frames not in (2,6):raise ValueError('Original startup/steady frame count required')
        h,w=self.working_hw
        return f'f{frames}_{w}x{h}'

    def reverse_windows(self,windows,frames):
        """Original WindowPartition3D.reverse, without a fixed spatial grid."""
        self.key(frames)
        h,w=self.token_hw
        if windows.shape!=((frames//2)*self.spatial_windows,128,1536):
            raise ValueError('Attention window geometry changed')
        value=windows.reshape(1,frames//2,h//8,w//8,2,8,8,1536)
        return np.ascontiguousarray(value.transpose(0,1,4,2,5,3,6,7).reshape(1,frames*h*w,1536))

    def prepare(self,path:Path):
        from PIL import Image
        with Image.open(path)as image:
            # Resize the original source directly: do not downsample it to a
            # model-specific quarter size or crop away any part of the frame.
            visible=np.array(image.convert('RGB').resize((self.width,self.height),Image.Resampling.BICUBIC),copy=True)
        h,w=self.working_hw
        return np.pad(visible,((0,h-self.height),(0,w-self.width),(0,0)),mode='edge')

    def write_windows(self,destination,windows,start,frames):
        """Place one completed query batch directly in original token order."""
        from window_pre import window_indices
        self.key(frames)
        if (destination.shape!=(1,frames*self.spatial_tokens,1536) or destination.dtype!=np.float16
                or windows.ndim!=3 or windows.shape[1:]!=(128,1536) or windows.dtype!=np.float16):
            raise ValueError('Attention destination/batch ABI changed')
        indices=window_indices((frames,*self.token_hw),start,len(windows))
        destination[0,indices]=windows.reshape(-1,1536)

    def trim(self,frame):
        if frame.shape[-2:]!=self.working_hw:raise ValueError('Unexpected working frame geometry')
        return frame[...,:self.height,:self.width]


def frame_pair_tiles(source_hw):
    """Return JSON-shaped full-width frame-pair tile geometry."""
    if (not isinstance(source_hw, (tuple, list)) or len(source_hw) != 2 or
            any(type(value) is not int for value in source_hw)):
        raise ValueError('frame-pair source geometry must be two integers')
    height, width = source_hw
    # FrameGeometry supplies the reviewed side/area bounds; retain the source
    # values below instead of silently replacing them with working_hw.
    FrameGeometry(height, width)
    if height < 32 or height % 32 or width % 8:
        raise ValueError('frame-pair source must have H>=32,H%32==0,W%8==0')
    middle = height // 2
    tile_height = middle + 16
    tile_middle = tile_height // 2
    if tile_height % 8 or tile_middle < 16:
        raise ValueError('frame-pair tile is not 8-aligned')
    tiles = [
        {'name': 'top', 'input_rows': (0, tile_middle),
         'output_rows': (0, middle), 'frame_rows': (0, middle)},
        {'name': 'bottom', 'input_rows': (middle - tile_middle, middle),
         'output_rows': (16, tile_height), 'frame_rows': (middle, height)},
    ]
    rows = [row for tile in tiles for row in range(*tile['frame_rows'])]
    if rows != list(range(height)):
        raise AssertionError('frame-pair tile rows are not an exact partition')
    return {
        'source_hw': (height, width), 'tile_hw': (tile_height, width),
        'middle_shape': (1, 128, tile_middle, width // 2),
        'tiles': tuple(tiles),
    }

def history_memory_tiles(source_hw, *, tile_count=4, halo_rows=16):
    """Return exact-core tiles for the fused history memory stages.

    Coordinates are in the history-stage middle raster: source rows are
    therefore twice the middle height, while frame_rows are middle-output
    rows.  The reviewed defaults retain a 16-row middle-raster halo; the
    explicit 16-lane option uses a 12-row halo and also exposes doubled
    source-frame scatter rows.  The first and last tile are clamped to the
    actual raster boundaries; no external zero padding is part of this
    geometry.
    """
    if (not isinstance(source_hw, (tuple, list)) or len(source_hw) != 2 or
            any(type(value) is not int for value in source_hw)):
        raise ValueError("history source geometry must be two integers")
    if type(tile_count) is not int or type(halo_rows) is not int:
        raise ValueError("history tile_count/halo_rows must be integers")
    if (tile_count, halo_rows) not in ((4, 16), (8, 16), (16, 12)):
        raise ValueError(
            "history geometry requires (tile_count, halo_rows) of "
            "(4,16), (8,16), or opt-in (16,12)"
        )
    height, width = source_hw
    FrameGeometry(height, width)
    if height < 128 or height % 32 or width % 8:
        raise ValueError(
            "history source must have H>=128 and an even compact tile partition"
        )
    middle = height // 2
    if middle % tile_count:
        raise ValueError("history middle raster is not evenly tile-partitioned")
    core = middle // tile_count
    halo = halo_rows
    if core < halo:
        raise ValueError("history tile core must cover its interior halo")
    tile_middle = core + 2 * halo
    if tile_middle > middle:
        raise ValueError("history tile exceeds the middle raster")

    names = (("top_outer", "top_inner", "bottom_inner", "bottom_outer")
             if tile_count == 4 else tuple(f"tile{index}" for index in range(tile_count)))
    tiles = []
    for index, name in enumerate(names):
        global_start = index * core
        global_stop = global_start + core
        input_start = min(max(global_start - halo, 0), middle - tile_middle)
        input_stop = input_start + tile_middle
        if input_start % 2 or input_stop % 2:
            raise ValueError("history compact boundaries must be even")
        output_start = global_start - input_start
        output_stop = global_stop - input_start
        output_rows = (output_start, output_stop)
        tile = {
            "name": name,
            "input_rows": (input_start, input_stop),
            "output_rows": output_rows,
            "frame_rows": (global_start, global_stop),
            "compact_rows": (input_start // 2, input_stop // 2),
        }
        if tile_count == 16:
            tile.update({
                "source_frame_rows": (2 * global_start, 2 * global_stop),
                "source_output_rows": (2 * output_start, 2 * output_stop),
            })
        tiles.append(tile)

    frame_rows = [
        row for tile in tiles for row in range(*tile["frame_rows"])
    ]
    if frame_rows != list(range(middle)):
        raise AssertionError("history tile cores are not an exact partition")
    for index, tile in enumerate(tiles):
        input_start, input_stop = tile["input_rows"]
        output_start, output_stop = tile["output_rows"]
        global_start, global_stop = tile["frame_rows"]
        if (output_start, output_stop) != (
                global_start - input_start, global_stop - input_start):
            raise AssertionError("history output rows lost their local offset")
        if index not in (0, len(tiles) - 1) and (
                global_start - input_start != halo or
                input_stop - global_stop != halo):
            raise AssertionError("interior history tile lost its halo")
        if index == 0 and (input_start != 0 or global_start != 0):
            raise AssertionError("top history tile is not at the outer boundary")
        if index == len(tiles) - 1 and (
                input_stop != middle or global_stop != middle):
            raise AssertionError("bottom history tile is not at the outer boundary")
    return {
        "source_hw": (height, width),
        "source_middle_shape": (1, 128, middle, width // 2),
        "source_compact_shape": (1, 256, height // 4, width // 4),
        "tile_source_hw": (2 * tile_middle, width),
        "tile_middle_shape": (1, 128, tile_middle, width // 2),
        "tile_compact_shape": (1, 256, tile_middle // 2, width // 4),
        "tile_count": tile_count,
        "halo_rows": halo,
        "tiles": tuple(tiles),
    }
