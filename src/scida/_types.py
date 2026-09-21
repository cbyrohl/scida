"""Type aliases for scida."""

from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Union

import dask.array as da
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from scida.fields import FieldContainer

# Path types
PathType = Union[str, PathLike[str]]

# Array types
ArrayLike = Union[np.ndarray, da.Array]
DaskArray = da.Array
NumpyArray = NDArray[Any]

# Field types
FieldFunction = Callable[["FieldContainer"], ArrayLike]

# Metadata types
MetadataDict = dict[str, Any]

# Chunksize type
ChunkSizeType = Union[str, int, tuple[int, ...]]
