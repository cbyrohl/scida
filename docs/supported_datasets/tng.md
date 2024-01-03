# The TNG Simulation Suite

## Overview
The IllustrisTNG project is a series of large-scale cosmological
magnetohydrodynamical simulations of galaxy formation. The data is
available at [www.tng-project.org](https://www.tng-project.org/).

## Demo data

Many of the examples in this documentation use the TNG50-4 simulation.
In particular, we make a snapshot and group catalog available to run
these examples. You can download and extract the snapshot and its group
catalog from the TNG50-4 test data:

``` bash
wget https://heibox.uni-heidelberg.de/f/dc65a8c75220477eb62d/?dl=1 -O snapshot.tar.gz
tar -xvf snapshot.tar.gz
wget https://heibox.uni-heidelberg.de/f/ff27fb6975fb4dc391ef/?dl=1 -O catalog.tar.gz
tar -xvf catalog.tar.gz
```

The snapshot and group catalog should be placed in the same folder.
Then you can load the snapshot with `ds = load("./snapdir_030")`. The group catalog should automatically be detected,
otherwise you can also pass the path to the catalog via the `catalog` keyword to `load()`.
