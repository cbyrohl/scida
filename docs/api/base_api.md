# Base API

## Convenience functions

Commonly used functions and classes.
There are two main object types in scida:

* [Dataset](moduleindex.md#scida.interface.Dataset)
* [DatasetSeries](moduleindex.md#scida.series.DatasetSeries)

All specialized classes derive from these two classes.
Generally, we do not have to instantiate the correct class ourselves,
but can use the [load](#scida.convenience.load) function to load a dataset.
This function will furthermore adjust the underlying class with additional
functionality, such as units, depending on the data set.

::: scida.convenience
    options:
      show_source: false
      members:
        - load


## Datasets and Series

The [Dataset](moduleindex.md#scida.interface.Dataset) class is the base class for all data sets.
Collections of datasets are represented by the [DatasetSeries](moduleindex.md#scida.series.DatasetSeries) class.
These objects are not instantiated directly, but are created by the [load](#scida.convenience.load) function.

For example, AREPO snapshots are defined by the [ArepoSnapshot](moduleindex.md#scida.customs.arepo.dataset.ArepoSnapshot) class.
The [load](#scida.convenience.load) function will select a class based on the following criteria (descending priority):

1. The class is passed to load as the force_class argument.
2. The class is specified in the simulation configuration, see [here](../configuration.md#simulation-configuration).
3. The class implements validate_path() returning True for the given path if the class is applicable.
