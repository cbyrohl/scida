# Supported datasets

The following table shows supported datasets.

| Name        | Support              |
| ----------- | -------------------- |
| [TNG](https://www.tng-project.org/)       | :material-check-all:     |
| [Illustris](https://www.illustris-project.org/)       | :material-check-all: |

A :material-check-all: checkmark indicates support out-of-the-box, a :material-check: checkmark indicates support by creating a suitable configuration file.


# File-format requirements

As of now, two underlying file formats are supported: hdf5 and zarr. Multi-file hdf5 is supported, for which a directory is passed as *path*, which contains only hdf5 files of the pattern *prefix.XXX.hdf5*, where *prefix* will be determined automatically and *XXX* is a contiguous list of integers indicating the order of hdf5 files to be merged. Hdf5 files are expected to have the same structure and all fields, i.e. hdf5 datasets, will be concatenated along their first axis.
