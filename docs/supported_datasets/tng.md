# The TNG Simulation Suite

## Overview
The IllustrisTNG project is a series of large-scale cosmological
magnetohydrodynamical simulations of galaxy formation. The data is
available at [www.tng-project.org](https://www.tng-project.org/).

## Demo data

Many of the examples in this documentation use the TNG50-4 simulation.
In particular, we make a snapshot and group catalog available to run
these examples. You can download and extract the snapshot and its group
catalog from the TNG50-4 test data using the following commands:

``` bash
wget https://heibox.uni-heidelberg.de/f/dc65a8c75220477eb62d/?dl=1 -O snapshot.tar.gz
tar -xvf snapshot.tar.gz
wget https://heibox.uni-heidelberg.de/f/ff27fb6975fb4dc391ef/?dl=1 -O catalog.tar.gz
tar -xvf catalog.tar.gz
```

These files are exactly [the same files](https://www.tng-project.org/api/TNG50-4/files/snapshot-30/)
that can be downloaded from the official IllustrisTNG data release.

The snapshot and group catalog should be placed in the same folder.
Then you can load the snapshot with `ds = load("./snapdir_030")`.
If you are executing the code from a different folder, you need to adjust the path accordingly.
The group catalog should automatically be detected when available in the same parent folder as the snapshot,
otherwise you can also pass the path to the catalog via the `catalog` keyword to `load()`.

## TNGLab

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

where "TNG50-4" is a pre-defined shortcut to the TNG50-4 simulation path on TNGLab. After having loaded the simulation, we request the snapshot "30" as used in the demo data. Custom shortcuts can be defined in the [simulation configuration](../configuration.md#simulation-configuration).
