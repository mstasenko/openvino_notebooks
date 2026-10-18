"""UHD-only CPU parser; current observations are not allocation reservations."""
import re

MIN_FREE = 12_400_000_000
MAX_ALLOCATED = 256 * 1024**2


def physical_admission(path, allocated, preserve):
    observation = {"torch_allocated_bytes": allocated, "minimum_free_bytes": MIN_FREE,
                   "maximum_torch_allocated_bytes": MAX_ALLOCATED,
                   "scope": "current driver observation, not physical peak or reservation"}
    try:
        raw = path.read_text()
    except OSError as error:
        observation.update(status="REJECTED", error=str(error))
        preserve(observation)
        raise
    observation["raw"] = raw
    fields = {key: re.findall(r"^\s*" + key + r":([^\n]*)$", raw, re.MULTILINE)
              for key in ("size", "usage")}
    error = None
    if any(len(values) != 1 or re.fullmatch(r"\s*[0-9]+\s*", values[0]) is None
           for values in fields.values()):
        error = "physical allocator size/usage missing or duplicated"
    else:
        size, usage = int(fields["size"][0]), int(fields["usage"][0])
        observation.update(size_bytes=size, usage_bytes=usage, free_bytes=size-usage)
        if not 0 <= usage <= size or size-usage < MIN_FREE:
            error = "UHD physical-free admission denied"
    if type(allocated) is not int or not 0 <= allocated <= MAX_ALLOCATED:
        error = "Torch allocated admission denied before UHD DiT upload"
    observation["status"] = "REJECTED" if error else "ACCEPTED"
    if error:
        observation["error"] = error
    preserve(observation)  # Retain raw/current values even on rejection.
    if error:
        raise ValueError(error)
    return observation
