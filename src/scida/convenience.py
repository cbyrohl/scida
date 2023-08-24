import copy
import hashlib
import logging
import os
import pathlib
import shutil
import sys
import tarfile
import time
from typing import Optional, Union

import requests

from scida.config import get_config
from scida.discovertypes import (
    _determine_mixins,
    _determine_type,
    _determine_type_from_simconfig,
)
from scida.interfaces.mixins import UnitMixin
from scida.io import load_metadata
from scida.registries import dataseries_type_registry, dataset_type_registry
from scida.series import DatasetSeries

log = logging.getLogger(__name__)


def download_and_extract(
    url: str, path: pathlib.Path, progressbar: bool = True, overwrite: bool = False
):
    """
    Download and extract a file from a given url.
    Parameters
    ----------
    url: str
        The url to download from.
    path: pathlib.Path
        The path to download to.
    progressbar: bool
        Whether to show a progress bar.
    overwrite: bool
        Whether to overwrite an existing file.
    Returns
    -------
    str
        The path to the downloaded and extracted file(s).
    """
    if path.exists() and not overwrite:
        raise ValueError("Target path '%s' already exists." % path)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        totlength = int(r.headers.get("content-length", 0))
        lread = 0
        t1 = time.time()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=2**22):  # chunks of 4MB
                t2 = time.time()
                f.write(chunk)
                lread += len(chunk)
                if progressbar:
                    rate = (lread / 2**20) / (t2 - t1)
                    sys.stdout.write(
                        "\rDownloaded %.2f/%.2f Megabytes (%.2f%%, %.2f MB/s)"
                        % (
                            lread / 2**20,
                            totlength / 2**20,
                            100.0 * lread / totlength,
                            rate,
                        )
                    )
                    sys.stdout.flush()
        sys.stdout.write("\n")
    tar = tarfile.open(path, "r:gz")
    for t in tar:
        if t.isdir():
            t.mode = int("0755", base=8)
        else:
            t.mode = int("0644", base=8)
    tar.extractall(path.parents[0])
    foldername = tar.getmembers()[0].name  # parent folder of extracted tar.gz
    tar.close()
    os.remove(path)
    return os.path.join(path.parents[0], foldername)


def get_testdata(name: str) -> str:
    """
    Get path to test data identifier.
    Parameters
    ----------
    name: str
        Name of test data.
    Returns
    -------
    str
    """
    config = get_config()
    tdpath = config.get("testdata_path", None)
    if tdpath is None:
        raise ValueError("Test data directory not specified in configuration")
    if not os.path.isdir(tdpath):
        raise ValueError("Invalid test data path")
    res = {f: os.path.join(tdpath, f) for f in os.listdir(tdpath)}
    if name not in res.keys():
        raise ValueError("Specified test data not available.")
    return res[name]


def find_path(path, overwrite=False) -> str:
    """
    Find path to dataset.
    Parameters
    ----------
    path: str
    overwrite: bool
        Only for remote datasets. Whether to overwrite an existing download.

    Returns
    -------
    str

    """
    config = get_config()
    path = os.path.expanduser(path)
    if os.path.exists(path):
        # datasets on disk
        pass
    elif len(path.split(":")) > 1:
        # check different alternative backends
        databackend = path.split("://")[0]
        dataname = path.split("://")[1]
        if databackend in ["http", "https"]:
            # dataset on the internet
            savepath = config.get("download_path", None)
            if savepath is None:
                print(
                    "Have not specified 'download_path' in config. Using 'cache_path' instead."
                )
                savepath = config.get("cache_path")
            savepath = os.path.expanduser(savepath)
            savepath = pathlib.Path(savepath)
            savepath.mkdir(parents=True, exist_ok=True)
            urlhash = str(
                int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % 10**8
            )
            savepath = savepath / ("download" + urlhash)
            filename = "archive.tar.gz"
            if not savepath.exists():
                os.makedirs(savepath, exist_ok=True)
            elif overwrite:
                # delete all files in folder
                for f in os.listdir(savepath):
                    fp = os.path.join(savepath, f)
                    if os.path.isfile(fp):
                        os.unlink(fp)
                    else:
                        shutil.rmtree(fp)
            foldercontent = [f for f in savepath.glob("*")]
            if len(foldercontent) == 0:
                savepath = savepath / filename
                extractpath = download_and_extract(
                    path, savepath, progressbar=True, overwrite=overwrite
                )
            else:
                extractpath = savepath
            extractpath = pathlib.Path(extractpath)

            # count folders in savepath
            nfolders = len([f for f in extractpath.glob("*") if f.is_dir()])
            nobjects = len([f for f in extractpath.glob("*") if f.is_dir()])
            if nobjects == 1 and nfolders == 1:
                extractpath = (
                    extractpath / [f for f in extractpath.glob("*") if f.is_dir()][0]
                )
            path = extractpath
        elif databackend == "testdata":
            path = get_testdata(dataname)
        else:
            # potentially custom dataset.
            resources = config.get("resources", {})
            if databackend not in resources:
                raise ValueError("Unknown resource '%s'" % databackend)
            r = resources[databackend]
            if dataname not in r:
                raise ValueError(
                    "Unknown dataset '%s' in resource '%s'" % (dataname, databackend)
                )
            path = os.path.expanduser(r[dataname]["path"])
    else:
        found = False
        if "datafolders" in config:
            for folder in config["datafolders"]:
                folder = os.path.expanduser(folder)
                if os.path.exists(os.path.join(folder, path)):
                    path = os.path.join(folder, path)
                    found = True
                    break
        if not found:
            raise ValueError("Specified path '%s' unknown." % path)
    return path


def load(
    path: str,
    units: Union[bool, str] = True,
    unitfile: str = "",
    overwrite: bool = False,
    force_class: Optional[object] = None,
    **kwargs
):
    path = find_path(path, overwrite=overwrite)

    if "catalog" in kwargs:
        c = kwargs["catalog"]
        query_path = True
        query_path &= c is not None
        query_path &= not isinstance(c, bool)
        query_path &= not isinstance(c, list)
        query_path &= c != "none"
        if query_path:
            kwargs["catalog"] = find_path(c, overwrite=overwrite)

    # determine dataset class
    reg = dict()
    reg.update(**dataset_type_registry)
    reg.update(**dataseries_type_registry)

    path = os.path.realpath(path)
    cls = _determine_type(path, **kwargs)[1][0]

    msg = "Dataset is identified as '%s' via _determine_type." % cls
    log.debug(msg)

    # any identifying metadata?
    classtype = "dataset"
    if issubclass(cls, DatasetSeries):
        classtype = "series"
    cls_simconf = _determine_type_from_simconfig(path, classtype=classtype, reg=reg)

    if cls_simconf and not issubclass(cls, cls_simconf):
        oldcls = cls
        cls = cls_simconf
        if oldcls != cls:
            msg = "Dataset is identified as '%s' via the simulation config replacing prior candidate '%s'."
            log.debug(msg % (cls, oldcls))
        else:
            msg = "Dataset is identified as '%s' via the simulation config, identical to prior candidate."
            log.debug(msg % cls)

    if force_class is not None:
        cls = force_class

    # determine additional mixins not set by class
    mixins = []
    if hasattr(cls, "_unitfile"):
        unitfile = cls._unitfile
    if unitfile:
        if not units:
            units = True
        kwargs["unitfile"] = unitfile

    if units:
        mixins.append(UnitMixin)
        kwargs["units"] = units

    kwargs["overwritecache"] = overwrite

    # we append since unit mixin is added outside of this func right now
    metadata_raw = dict()
    if classtype == "dataset":
        metadata_raw = load_metadata(path, fileprefix=None)
    other_mixins = _determine_mixins(path=path, metadata_raw=metadata_raw)
    mixins += other_mixins

    log.debug("Adding mixins '%s' to dataset." % mixins)
    if hasattr(cls, "_mixins"):
        cls_mixins = cls._mixins
        for m in cls_mixins:
            # remove duplicates
            if m in mixins:
                mixins.remove(m)

    instance = cls(path, mixins=mixins, **kwargs)
    return instance


def get_dataset_by_name(name: str) -> Optional[str]:
    """
    Get dataset name from alias or name found in the configuration files.
    Parameters
    ----------
    name: str
        Name or alias of dataset.

    Returns
    -------
    str
    """
    dname = None
    c = get_config()
    if "datasets" not in c:
        return dname
    datasets = copy.deepcopy(c["datasets"])
    if name in datasets:
        dname = name
    else:
        # could still be an alias
        for k, v in datasets.items():
            if "aliases" not in v:
                continue
            if name in v["aliases"]:
                dname = k
                break
    return dname


def get_datasets_by_props(**kwargs):
    dnames = []
    c = get_config()
    if "datasets" not in c:
        return dnames
    datasets = copy.deepcopy(c["datasets"])
    for k, v in datasets.items():
        props = v.get("properties", {})
        match = True
        for pk, pv in kwargs.items():
            if pk not in props:
                match = False
                break
            if props[pk] != pv:
                match = False
                break
        if match:
            dnames.append(k)
    return dnames


def get_dataset_candidates(name=None, props=None):
    if name is not None:
        dnames = [get_dataset_by_name(name)]
        return dnames
    if props is not None:
        dnames = get_datasets_by_props(**props)
        return dnames
    raise ValueError("Need to specify name or properties.")


def get_dataset(name=None, props=None):
    dnames = get_dataset_candidates(name=name, props=props)
    if len(dnames) > 1:
        raise ValueError("Too many dataset candidates.")
    elif len(dnames) == 0:
        raise ValueError("No dataset candidate found.")
    return dnames[0]
