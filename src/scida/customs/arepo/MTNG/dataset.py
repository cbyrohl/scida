import os
from typing import Union

from scida.customs.arepo.dataset import ArepoCatalog, ArepoSnapshot
from scida.discovertypes import CandidateStatus
from scida.io import load_metadata


class MTNGArepoSnapshot(ArepoSnapshot):
    _fileprefix_catalog = "fof_subhalo_tab"
    _fileprefix = "snapshot_"  # underscore is important!

    def __init__(self, path, chunksize="auto", catalog=None, **kwargs) -> None:
        tkwargs = dict(
            fileprefix=self._fileprefix,
            fileprefix_catalog=self._fileprefix_catalog,
            catalog_cls=MTNGArepoCatalog,
        )
        tkwargs.update(**kwargs)

        # in MTNG, we have two kinds of snapshots:
        # 1. regular (prefix: "snapshot_"), contains all particle types
        # 2. mostbound (prefix: "snapshot-prevmostboundonly_"), only contains DM particles
        # Most snapshots are of type 2, but some selected snapshots have type 1 and type 2.

        # attempt to load regular snapshot
        super().__init__(path, chunksize=chunksize, catalog=catalog, **tkwargs)
        # later unit file takes precedence
        self._defaultunitfiles += ["units/mtng.yaml"]
        self.parameters = {}
        self._grouplengths = {}
        self._subhalolengths = {}
        # not needed for group catalogs as entries are back-to-back there, we will provide a property for this
        self._subhalooffsets = {}
        self.misc = {}  # for storing misc info
        prfx = kwargs.pop("fileprefix", None)
        if prfx is None:
            prfx = self._get_fileprefix(path)
        super().__init__(path, chunksize=chunksize, fileprefix=prfx, **kwargs)

        self.catalog = catalog
        if not self.iscatalog:
            if self.catalog is None:
                self.discover_catalog()
                # try to discover group catalog in parent directories.
            if self.catalog == "none":
                pass  # this string can be set to explicitly disable catalog
            elif self.catalog:
                catalog_cls = kwargs.get("catalog_cls", None)
                self.load_catalog(kwargs, catalog_cls=catalog_cls)

        # add aliases
        aliases = dict(
            PartType0=["gas", "baryons"],
            PartType1=["dm", "dark matter"],
            PartType2=["lowres", "lowres dm"],
            PartType3=["tracer", "tracers"],
            PartType4=["stars"],
            PartType5=["bh", "black holes"],
        )
        for k, lst in aliases.items():
            if k not in self.data:
                continue
            for v in lst:
                self.data.add_alias(v, k)

        # set metadata
        self._set_metadata()

        if tkwargs["fileprefix"] == "snapshot-prevmostboundonly_":
            # this is a mostbound snapshot, so we are done
            return

        # next, attempt to load mostbound snapshot. This is done by loading into sub-object.
        self.mostbound = None
        tkwargs.update(fileprefix="snapshot-prevmostboundonly_", catalog="none")
        self.mostbound = MTNGArepoSnapshot(path, chunksize=chunksize, **tkwargs)
        # hacky: remove unused containers from mostbound snapshot
        for k in [
            "PartType0",
            "PartType2",
            "PartType3",
            "PartType4",
            "PartType5",
            "Group",
            "Subhalo",
        ]:
            if k in self.mostbound.data:
                del self.mostbound.data[k]
        self.merge_data(self.mostbound, fieldname_suffix="_mostbound")

    @classmethod
    def validate_path(
        cls, path: Union[str, os.PathLike], *args, **kwargs
    ) -> CandidateStatus:
        tkwargs = dict(
            fileprefix=cls._fileprefix, fileprefix_catalog=cls._fileprefix_catalog
        )
        tkwargs.update(**kwargs)
        try:
            valid = super().validate_path(path, *args, **tkwargs)
        except ValueError:
            valid = CandidateStatus.NO
            # might raise ValueError in case of partial snap

        if valid == CandidateStatus.NO:
            # check for partial snap
            tkwargs.update(fileprefix="snapshot-prevmostboundonly_")
            try:
                valid = super().validate_path(path, *args, **tkwargs)
            except ValueError:
                valid = CandidateStatus.NO

        if valid == CandidateStatus.NO:
            return valid
        metadata_raw = load_metadata(path, **tkwargs)
        if "/Config" not in metadata_raw:
            return CandidateStatus.NO
        if "MTNG" not in metadata_raw["/Config"]:
            return CandidateStatus.NO
        if "Ngroups_Total" in metadata_raw["/Header"]:
            return CandidateStatus.NO  # this is a catalog
        return CandidateStatus.YES


class MTNGArepoCatalog(ArepoCatalog):
    _fileprefix = "fof_subhalo_tab"

    def __init__(self, *args, **kwargs):
        kwargs["iscatalog"] = True
        if "fileprefix" not in kwargs:
            kwargs["fileprefix"] = "fof_subhalo_tab"
        kwargs["choose_prefix"] = True
        super().__init__(*args, **kwargs)

    @classmethod
    def validate_path(
        cls, path: Union[str, os.PathLike], *args, **kwargs
    ) -> CandidateStatus:
        tkwargs = dict(fileprefix=cls._fileprefix)
        tkwargs.update(**kwargs)
        valid = super().validate_path(path, *args, **tkwargs)
        if valid == CandidateStatus.NO:
            return valid
        metadata_raw = load_metadata(path, **tkwargs)
        if "/Config" not in metadata_raw:
            return CandidateStatus.NO
        if "MTNG" not in metadata_raw["/Config"]:
            return CandidateStatus.NO
        return CandidateStatus.YES
