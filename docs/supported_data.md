# Supported datasets

The following table shows a selection of supported datasets. The table is not exhaustive, but should give an idea of the range of supported datasets.
If you want to use a dataset that is not listed here, read on [here](dataset_structure.md) and consider opening an issue or contact us directly.

| Name                                                  | Support                               | Description                                                                                                     |
|-------------------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [AURIGA](https://wwwmpa.mpa-garching.mpg.de/auriga/)  | :material-check-all:                  | Cosmological zoom-in galaxy formation *simulations*                                                             |
| [EAGLE](https://icc.dur.ac.uk/Eagle/)                 | :material-check-all:                  | Cosmological galaxy formation *simulations*                                                                     |
| [FIRE2](https://wetzel.ucdavis.edu/fire-simulations/) | :material-check-all:                  | Cosmological zoom-in galaxy formation *simulations*                                                             |
| [FLAMINGO](https://flamingo.strw.leidenuniv.nl/)      | :material-check-all:                  | Cosmological galaxy formation *simulations*                                                                     |
| [Gaia](https://www.cosmos.esa.int/web/gaia/dr3)       | :material-database-check-outline:[^1] | *Observations* of a billion nearby stars                                                                        |
| [Illustris](https://www.illustris-project.org/)       | :material-check-all:                  | Cosmological galaxy formation *simulations*                                                                     |
| [LGalaxies](customs/lgalaxies.md)                     | :material-check-all:                  | Semi-analytical model for [Millenium](https://wwwmpa.mpa-garching.mpg.de/galform/virgo/millennium/) simulations |
| [SDSS DR16](https://www.sdss.org/dr16/)               | :material-check:                      | *Observations* for millions of galaxies                                                                         |
| [SIMBA](http://simba.roe.ac.uk/)                      | :material-check-all:                  | Cosmological galaxy formation *simulations*                                                                     |
| [TNG](https://www.tng-project.org/)                   | :material-check-all:                  | Cosmological galaxy formation *simulations*                                                                     |
| TNG-Cluster                                           | :material-check-all:                  | Cosmological zoom-in galaxy formation *simulations*                                                             |



A :material-check-all: checkmark indicates support out-of-the-box, a :material-check: checkmark indicates work-in-progress support or the need to create a suitable configuration file.
A :material-database-check-outline: checkmark indicates support for converted HDF5 versions of the original data.


# File-format requirements

As of now, two underlying file formats are supported: hdf5 and zarr. Multi-file hdf5 is supported, for which a directory is passed as *path*, which contains only hdf5 files of the pattern *prefix.XXX.hdf5*, where *prefix* will be determined automatically and *XXX* is a contiguous list of integers indicating the order of hdf5 files to be merged. Hdf5 files are expected to have the same structure and all fields, i.e. hdf5 datasets, will be concatenated along their first axis.

Support for FITS is work-in-progress, also see [here](../tutorial/observations/#fits-files) for a proof-of-concept.

[^1]: The HDF5 version of GAIA DR3 is available [here](https://www.tng-project.org/data/obs/).
