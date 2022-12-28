from astrodask.interfaces.arepo import ArepoSnapshot


class IllustrisSnapshot(ArepoSnapshot):
    def __init__(self, path, chunksize="auto", catalog=None, **kwargs):
        super().__init__(
            path,
            chunksize=chunksize,
            catalog=catalog,
            catalog_kwargs=dict(fileprefix="groups"),
            **kwargs
        )
