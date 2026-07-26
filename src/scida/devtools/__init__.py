"""
Developer utilities for deploying pre-cache files.

Provides functions to copy local cache files into a shared basefolder
(default ``~/scida_precache_files/postprocessing/scida/``) so that all
users sharing the same dataset tree benefit from a pre-built cache.

Can also be invoked as a CLI::

    python -m scida.devtools build /path/to/simulation
    python -m scida.devtools deploy-all
"""

from scida.devtools._cache_deploy import (
    MPCDF_TARGETS,
    deploy_precache,
    deploy_series_precache,
)

__all__ = [
    "MPCDF_TARGETS",
    "deploy_precache",
    "deploy_series_precache",
]
