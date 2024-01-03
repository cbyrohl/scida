# Base API

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
