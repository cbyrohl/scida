import os
from typing import Union

from scida import GadgetStyleSnapshot
from scida.io import load_metadata


class SwiftSnapshot(GadgetStyleSnapshot):
    def __init__(self, path, chunksize="auto", virtualcache=True, **kwargs) -> None:
        super().__init__(path, chunksize=chunksize, virtualcache=virtualcache, **kwargs)

    @classmethod
    def validate_path(cls, path: Union[str, os.PathLike], *args, **kwargs) -> bool:
        valid = super().validate_path(path, *args, **kwargs)
        if not valid:
            return False
        metadata_raw = load_metadata(path, **kwargs)
        comparestr = metadata_raw.get("/Code", {}).get("Code", b"").decode()
        valid = "SWIFT" in comparestr
        return valid
