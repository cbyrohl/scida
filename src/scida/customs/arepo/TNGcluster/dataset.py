from scida import ArepoSnapshot


class TNGClusterSnapshot(ArepoSnapshot):
    _fileprefix_catalog = "fof_subhalo_tab"
    _fileprefix = "snapshot_"  # underscore is important!
