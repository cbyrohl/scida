# Series

!!! info


    If you want to run the code below, you need a folder containing multiple scida datasets as subfolders.
    Specify the path in load() to the base directory of the series.
    The example below uses an AREPO simulation, the TNG50-4 simulation, as a series of snapshots.
    This simulation can be downloaded from the [TNG website](https://www.tng-project.org/data/)
    or directly accessed online in the [TNGLab](https://www.tng-project.org/data/lab/).

In the tutorial section, we have only considered individual data sets.
Often data sets are given in a series (e.g. multiple snapshots of a simulation, multiple exposures in a survey).
Loading this as a series provides convenient access to all contained objects.

``` pycon
>>> from scida import load
>>> series = load("TNG50-4") #(1)!
```

1. Pass the base path of the simulation.

We can now access the individual data sets from the series object:

``` pycon
>>> series[0] #(1)!
```

1. Alias for 'series.datasets[0]'

Depending on the available metadata, we can select data sets by these.

For example, cosmological simulations usually have information about their redshift:

``` pycon
>>> snp = series.get_dataset(redshift=2.0)
>>> snp.header["Redshift"]
2.0020281392528516
```
