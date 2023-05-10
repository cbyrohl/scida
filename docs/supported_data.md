TODO

As of now, two underlying file formats are supported: hdf5 and zarr. Multi-file hdf5 is supported, for which a directory is passed as *path*, which contains only hdf5 files of the pattern *prefix.XXX.hdf5*, where *prefix* will be determined automatically and *XXX* is a contiguous list of integers indicating the order of hdf5 files to be merged. Hdf5 files are expected to have the same structure and all fields, i.e. hdf5 datasets, will be concatenated along their first axis.
