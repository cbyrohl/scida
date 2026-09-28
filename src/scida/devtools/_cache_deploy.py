"""
Developer utilities for deploying pre-cache files.

Provides functions to copy local cache files into a shared basefolder
(default ``~/scida_precache_files/postprocessing/scida/``) so that all
users sharing the same dataset tree benefit from a pre-built cache.

"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil

from scida.config import get_config
from scida.helpers_misc import hash_path
from scida.misc import return_cachefile_path, return_hdf5cachepath

log = logging.getLogger(__name__)

# MPCDF simulation base paths for pre-cache deployment.
MPCDF_TARGETS = [
    # Millennium
    "/virgotng/universe/Millennium-1",
    "/virgotng/universe/Millennium-2",
    # Illustris
    "/virgotng/universe/Illustris/L75n455DM",
    "/virgotng/universe/Illustris/L75n455FP",
    "/virgotng/universe/Illustris/L75n455NR",
    "/virgotng/universe/Illustris/L75n910DM",
    "/virgotng/universe/Illustris/L75n910FP",
    "/virgotng/universe/Illustris/L75n910NR",
    "/virgotng/universe/Illustris/L75n1820DM",
    "/virgotng/universe/Illustris/L75n1820FP",
    # IllustrisTNG
    "/virgotng/universe/IllustrisTNG/L35n270TNG",
    "/virgotng/universe/IllustrisTNG/L35n270TNG_DM",
    "/virgotng/universe/IllustrisTNG/L35n540TNG",
    "/virgotng/universe/IllustrisTNG/L35n540TNG_DM",
    "/virgotng/universe/IllustrisTNG/L35n1080TNG",
    "/virgotng/universe/IllustrisTNG/L35n1080TNG_DM",
    "/virgotng/universe/IllustrisTNG/L35n2160TNG",
    "/virgotng/universe/IllustrisTNG/L35n2160TNG_DM",
    "/virgotng/universe/IllustrisTNG/L75n455TNG",
    "/virgotng/universe/IllustrisTNG/L75n455TNG_DM",
    "/virgotng/universe/IllustrisTNG/L75n455TNG_NR",
    "/virgotng/universe/IllustrisTNG/L75n455TNG_UVB",
    "/virgotng/universe/IllustrisTNG/L75n910TNG",
    "/virgotng/universe/IllustrisTNG/L75n910TNG_DM",
    "/virgotng/universe/IllustrisTNG/L75n910TNG_NR",
    "/virgotng/universe/IllustrisTNG/L75n1820TNG",
    "/virgotng/universe/IllustrisTNG/L75n1820TNG_DM",
    "/virgotng/universe/IllustrisTNG/L75n1820TNG_NR",
    "/virgotng/universe/IllustrisTNG/L205n625TNG",
    "/virgotng/universe/IllustrisTNG/L205n625TNG_DM",
    "/virgotng/universe/IllustrisTNG/L205n1250TNG",
    "/virgotng/universe/IllustrisTNG/L205n1250TNG_DM",
    "/virgotng/universe/IllustrisTNG/L205n2500TNG",
    "/virgotng/universe/IllustrisTNG/L205n2500TNG_DM",
    # AIDA-TNG
    "/virgotng/universe/AIDA-TNG/L35n540_CDM",
    "/virgotng/universe/AIDA-TNG/L35n540_SIDM1",
    "/virgotng/universe/AIDA-TNG/L35n540_SIDM1-Dark",
    "/virgotng/universe/AIDA-TNG/L35n540_vSIDM",
    "/virgotng/universe/AIDA-TNG/L35n540_vSIDM-Dark",
    "/virgotng/universe/AIDA-TNG/L35n540_WDM1",
    "/virgotng/universe/AIDA-TNG/L35n540_WDM1-Dark",
    "/virgotng/universe/AIDA-TNG/L35n540_WDM3",
    "/virgotng/universe/AIDA-TNG/L35n540_WDM3-Dark",
    "/virgotng/universe/AIDA-TNG/L35n1080_CDM",
    "/virgotng/universe/AIDA-TNG/L35n1080_SIDM1",
    "/virgotng/universe/AIDA-TNG/L35n1080_SIDM1-Dark",
    "/virgotng/universe/AIDA-TNG/L35n1080_vSIDM",
    "/virgotng/universe/AIDA-TNG/L35n1080_vSIDM-Dark",
    "/virgotng/universe/AIDA-TNG/L35n1080_WDM3",
    "/virgotng/universe/AIDA-TNG/L75n455_CDM",
    "/virgotng/universe/AIDA-TNG/L75n455_SIDM1",
    "/virgotng/universe/AIDA-TNG/L75n455_SIDM1-Dark",
    "/virgotng/universe/AIDA-TNG/L75n455_vSIDM",
    "/virgotng/universe/AIDA-TNG/L75n455_vSIDM-Dark",
    "/virgotng/universe/AIDA-TNG/L75n455_WDM1",
    "/virgotng/universe/AIDA-TNG/L75n455_WDM1-Dark",
    "/virgotng/universe/AIDA-TNG/L75n455_WDM3-Dark",
    "/virgotng/universe/AIDA-TNG/L75n910_CDM",
    "/virgotng/universe/AIDA-TNG/L75n910_SIDM1",
    "/virgotng/universe/AIDA-TNG/L75n910_vSIDM",
    "/virgotng/universe/AIDA-TNG/L75n910_WDM3",
    # Eagle
    "/virgotng/universe/Eagle/L68n1504DM",
    "/virgotng/universe/Eagle/L68n1504FP",
    "/virgotng/universe/Eagle/RecalL0025N0752",
    # Simba
    "/virgotng/universe/Simba/L25n512FP",
    "/virgotng/universe/Simba/L50n512FP",
    "/virgotng/universe/Simba/L100n1024FP",
    # FLAMINGO
    "/virgotng/mpia/FLAMINGO/L1000N0900",
]


def _detect_fileprefix(path):
    """
    Auto-detect the file prefix for a dataset path.

    Uses the type detection machinery to find the dataset class and
    calls its ``_get_fileprefix`` method.

    Parameters
    ----------
    path : str
        Resolved path to the dataset.

    Returns
    -------
    str
        The detected file prefix (may be empty string).
    """
    from scida.discovertypes import _determine_type

    try:
        names, types = _determine_type(
            path, test_datasets=True, test_dataseries=False, catch_exception=True
        )
    except ValueError:
        log.debug("Could not determine type for '%s', using empty prefix.", path)
        return ""
    for cls in types:
        if hasattr(cls, "_get_fileprefix"):
            return cls._get_fileprefix(path)
    return ""


def deploy_precache(
    paths, basefolder=None, fileprefix=None, overwrite=False, depth=2
):
    """
    Deploy local HDF5 cache files to a shared basefolder.

    For each path, finds the existing local cache file and copies it to
    ``{basefolder}/{ancestor}/postprocessing/scida/{hash}.hdf5``, where
    *ancestor* is determined by going *depth* levels up from the dataset
    path.  This mirrors the real filesystem so that copying the basefolder
    contents to ``/`` places every file where ``find_precached_file``
    expects it.

    Parameters
    ----------
    paths : str, pathlib.Path, or list thereof
        One or more dataset paths whose local caches should be deployed.
    basefolder : str or None, optional
        Target basefolder. Defaults to the ``precache_path`` config option
        (default ``~/scida_precache_files``).
    fileprefix : str or None, optional
        File prefix used for caching. If ``None``, auto-detected per path.
    overwrite : bool, optional
        If ``True``, replace existing deployed files (default ``False``).
    depth : int, optional
        Number of parent levels to go up from the dataset path to
        determine the ancestor directory (default ``2``, matching
        ``{sim}/output/snapdir_*`` → ``{sim}``).

    Returns
    -------
    list of str
        Paths to the deployed cache files.

    Raises
    ------
    FileNotFoundError
        If no local cache exists for a given path.
    FileExistsError
        If the target file already exists and *overwrite* is ``False``.
    """
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]

    if basefolder is None:
        config = get_config()
        basefolder = config.get("precache_path", "~/scida_precache_files")
    basefolder = os.path.expanduser(basefolder)

    deployed = []
    for p in paths:
        p = os.path.realpath(str(p))

        if fileprefix is not None:
            prefix = fileprefix
        else:
            prefix = _detect_fileprefix(p)

        local_cache = return_hdf5cachepath(p, fileprefix=prefix if prefix else None)
        if local_cache is None or not os.path.isfile(local_cache):
            raise FileNotFoundError(
                "No local cache found for '%s' (prefix='%s')." % (p, prefix)
            )

        hash_input = os.path.join(p, prefix) if prefix else p
        hsh = hash_path(hash_input)

        # Go up `depth` levels to find the ancestor directory
        ancestor = pathlib.Path(p)
        for _ in range(depth):
            ancestor = ancestor.parent
        dest_dir = os.path.join(
            basefolder, str(ancestor).lstrip("/"), "postprocessing", "scida"
        )
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, "%s.hdf5" % hsh)

        if os.path.exists(dest) and not overwrite:
            raise FileExistsError(
                "Deployed cache already exists: %s" % dest
            )

        shutil.copy2(local_cache, dest)
        deployed.append(dest)

    return deployed


def deploy_series_precache(paths, basefolder=None, overwrite=False, depth=3):
    """
    Deploy a local series (JSON) cache file to a shared basefolder.

    The file is placed at
    ``{basefolder}/{ancestor}/postprocessing/scida/{hash}.json``, where
    *ancestor* is determined by going *depth* levels up from the first
    path in the series.

    Parameters
    ----------
    paths : list of str or pathlib.Path
        Ordered list of dataset paths forming the series.
    basefolder : str or None, optional
        Target basefolder. Defaults to the ``precache_path`` config option
        (default ``~/scida_precache_files``).
    overwrite : bool, optional
        If ``True``, replace an existing deployed file (default ``False``).
    depth : int, optional
        Number of parent levels to go up from the first path to
        determine the ancestor directory (default ``3``, matching
        ``{sim}/output/snapdir_*/file.hdf5`` → ``{sim}``).

    Returns
    -------
    str
        Path to the deployed cache file.

    Raises
    ------
    FileNotFoundError
        If no local series cache exists.
    FileExistsError
        If the target file already exists and *overwrite* is ``False``.
    """
    if not paths:
        raise ValueError("paths must not be empty.")
    paths = [os.path.realpath(str(p)) for p in paths]

    if basefolder is None:
        config = get_config()
        basefolder = config.get("precache_path", "~/scida_precache_files")
    basefolder = os.path.expanduser(basefolder)

    hsh = hash_path("".join(paths))
    local_cache = return_cachefile_path(os.path.join(hsh, "data.json"))
    if local_cache is None or not os.path.isfile(local_cache):
        raise FileNotFoundError(
            "No local series cache found (hash=%s)." % hsh
        )

    # Go up `depth` levels from the first path to find the ancestor
    ancestor = pathlib.Path(paths[0])
    for _ in range(depth):
        ancestor = ancestor.parent
    dest_dir = os.path.join(
        basefolder, str(ancestor).lstrip("/"), "postprocessing", "scida"
    )
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "%s.json" % hsh)

    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(
            "Deployed series cache already exists: %s" % dest
        )

    shutil.copy2(local_cache, dest)
    return dest
