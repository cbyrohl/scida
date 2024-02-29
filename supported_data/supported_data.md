# Supported datasets

The following table shows a selection of supported datasets. The table is not exhaustive, but should give an idea of the range of supported datasets.
If you want to use a dataset that is not listed here, read on [here](#supported-file-formats-and-their-structure) and consider opening an issue or contact us directly.

| Name                                                                                  | Support                                                                                           | Description                                                                                                     |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [AURIGA](https://wwwmpa.mpa-garching.mpg.de/auriga/)                                  | :material-check-all:                                                                              | Cosmological zoom-in galaxy formation *simulations*                                                             |
| [EAGLE](https://icc.dur.ac.uk/Eagle/)                                                 | :material-check-all:                                                                              | Cosmological galaxy formation *simulations*                                                                     |
| [FIRE2](https://wetzel.ucdavis.edu/fire-simulations/)                                 | :material-check-all:                                                                              | Cosmological zoom-in galaxy formation *simulations*                                                             |
| [FLAMINGO](https://flamingo.strw.leidenuniv.nl/)                                      | :material-check-all:                                                                              | Cosmological galaxy formation *simulations*                                                                     |
| [Gaia](https://www.cosmos.esa.int/web/gaia/dr3)                                       | :material-database-check-outline:<sup>[\[download\]](https://www.tng-project.org/data/obs/)</sup> | *Observations* of a billion nearby stars                                                                        |
| [Illustris](https://www.illustris-project.org/)                                       | :material-check-all:                                                                              | Cosmological galaxy formation *simulations*                                                                     |
| [LGalaxies](https://lgalaxiespublicrelease.github.io/) <sup>[\[1\]](#lgalaxies)</sup> | :material-check-all:                                                                              | Semi-analytical model for [Millenium](https://wwwmpa.mpa-garching.mpg.de/galform/virgo/millennium/) simulations |
| [SDSS DR16](https://www.sdss.org/dr16/)                                               | :material-check:                                                                                  | *Observations* for millions of galaxies                                                                         |
| [SIMBA](http://simba.roe.ac.uk/)                                                      | :material-check-all:                                                                              | Cosmological galaxy formation *simulations*                                                                     |
| [TNG](https://www.tng-project.org/)<sup>[\[2\]](#the-tng-simulation-suite)</sup>                                  | :material-check-all:                                                                              | Cosmological galaxy formation *simulations*                                                                     |
| [TNG-Cluster](https://www.tng-project.org/cluster/)                                   | :material-check-all:                                                                              | Cosmological zoom-in galaxy formation *simulations*                                                             |


A :material-check-all: checkmark indicates support out-of-the-box, a :material-check: checkmark indicates work-in-progress support or the need to create a suitable configuration file.
A :material-database-check-outline: checkmark indicates support for converted HDF5 versions of the original data.

## Dataset Details

### LGalaxies

Access via individual datasets are supported, e.g.:

```pycon
>>> from scida import load
>>> load("LGal_Ayromlou2021_snap58.hdf5")
```

while access to the series at once (i.e. loading all data for all snapshots in a folder) is **not supported**.


### The TNG Simulation Suite

#### Overview
The IllustrisTNG project is a series of large-scale cosmological
magnetohydrodynamical simulations of galaxy formation. The data is
available at [www.tng-project.org](https://www.tng-project.org/).

#### Demo data

Many of the examples in this documentation use the TNG50-4 simulation.
In particular, we make a snapshot and group catalog available to run
these examples. You can download and extract the snapshot and its group
catalog from the TNG50-4 test data using the following commands:

``` bash
wget "https://heibox.uni-heidelberg.de/f/dc65a8c75220477eb62d/?dl=1" -O snapshot.tar.gz
tar -xvf snapshot.tar.gz
wget "https://heibox.uni-heidelberg.de/f/ff27fb6975fb4dc391ef/?dl=1" -O catalog.tar.gz
tar -xvf catalog.tar.gz
```

These files are exactly [the same files](https://www.tng-project.org/api/TNG50-4/files/snapshot-30/)
that can be downloaded from the official IllustrisTNG data release.

The snapshot and group catalog should be placed in the same folder.
Then you can load the snapshot with `ds = load("./snapdir_030")`.
If you are executing the code from a different folder, you need to adjust the path accordingly.
The group catalog should automatically be detected when available in the same parent folder as the snapshot,
otherwise you can also pass the path to the catalog via the `catalog` keyword to `load()`.

#### TNGLab

The [TNGLab](https://www.tng-project.org/data/lab/) is a web-based analysis platform running a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) instance with access to dedicated computational resources and all TNG data sets to provide
a convenient way to run analysis code on the TNG data sets. As TNGLab supports scida, it is a great way to get started and for running the examples.

In order to run the examples which use the [demo data](#demo-data), replace

``` py
ds = load("./snapdir_030")
```

with

``` py
ds = load("/home/tnguser/sims.TNG/TNG50-4/output/snapdir_030")
```

for these examples.

Alternatively, you can use

``` py
sim = load("TNG50-4")
ds = sim.get_dataset(30)
```

where "TNG50-4" is a pre-defined shortcut to the TNG50-4 simulation path on TNGLab. After having loaded the simulation, we request the snapshot "30" as used in the demo data. Custom shortcuts can be defined in the [simulation configuration](configuration.md#simulation-configuration).




## Supported file formats and their structure

Here, we discuss the requirements for easy extension/support of new datasets.
Currently, input files need to have one of the following formats:

* [hdf5](https://www.hdfgroup.org/solutions/hdf5/)
* multi-file hdf5: We assume a directory containing hdf5 files of the pattern *prefix.XXX.hdf5*, where *prefix* will be determined automatically and *XXX* is a contiguous list of integers indicating the order of hdf5 files to be merged. Hdf5 files are expected to have the same structure and all fields, i.e. hdf5 datasets, will be concatenated along their first axis.
* [zarr](https://zarr.readthedocs.io/en/stable/)

Support for FITS is work-in-progress, also see [here](tutorial/observations.md#fits-files) for a proof-of-concept.


Scida and above file formats use a hierarchical structure to store data with three fundamental objects:

* **Groups** are containers for other groups or datasets.
* **Datasets** are multidimensional arrays of a homogeneous type, usually bundled into some *Group*.
* **Attributes** provide various metadata.

At this point, we only support unstructured datasets, i.e. datasets that do not depend on the memory layout for their
interpretation. For example, this implies that simulation codes utilizing uniform or adaptive grids are not supported.

We explicitly support simulations run with the following codes:

* [Gadget](https://wwwmpa.mpa-garching.mpg.de/gadget4/)
* [Gizmo](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html)
* [Arepo](https://arepo-code.org/)
* [Swift](https://swift.strw.leidenuniv.nl/)
