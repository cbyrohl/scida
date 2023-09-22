---
title: 'scida: A Python package for big scientific data analysis'
tags:
  - Python
  - simulations
  - i/o
  - point clouds
authors:
  - name: Chris Byrohl
    orcid: 0000-0002-0885-8090
    affiliation: 1
  - name: Dylan Nelson
    orcid: 0000-0001-8421-5890
    affiliation: 1
affiliations:
 - name: Heidelberg University, Institute for Theoretical Astronomy, Albert-Ueberle-Str. 2, 69120 Heideberg, Germany
   index: 1
date: 15 Sep 2023
bibliography: paper.bib
---

# Summary

"scida" is a Python package for reading and processing large scientific data sets, utilizing the dask library
for scalable, parallel and out-of-core computation [@Dask].
Data access is provided in a hierarchical dictionary-like data attribute for datasets initialized via the load() function.
From there, any available dask (array) operation can be performed for the analysis. Computation requests issued in a
user session are collected in a task graph, that can be complemented with additional operations. The computation is
only executed on a target resource (e.g. a HPC cluster) away from the user session upon explicit request (see Figure \ref{fig:sketch}).

![Schematic of the workflow. A dataset object is initialized with scida.load(), holding a virtual reference to
 the underlying data. Operations are collected into a task graph via the dask library. Only upon issuing a compute()
command, the computation takes place on a target resource that can differ from the user session.
Results are substantially smaller than the original data set, and can be send back to the user session for further
analysis/plotting. \label{fig:sketch}](sketch.pdf)


# Features

Beside providing a dictionary-like interface to the underlying data as dask task graphs, scida provides a number of
additional features. Unit support is provided by the pint package [@Pint] and attached to the data from the metadata
where possible or manually specified by the user using configuration files. Scida attempts to automatically determine
the given data set by choosing an appropriate (i) file reader, (ii) instantiated subclass, and (iii) appropriate
features based on the data set. At the moment, file readers exist for zarr, single/multi-file HDF5 and FITS files.
Subclasses can add additional functionality for a given data set, such as halo selection for cosmological simulations.
Additional features are added as mixin classes, such as for unit support and spatial relations for data sets containing
cartesian coordinates.


# Statement of need
Today, scientific datasets often span terabytes in size. This complicates much of the analysis process away from
the scientific question at hand to data management and file handling, creating (i) a barrier for new researchers,
(ii) substantial time commitment away from the scientific endeavour, and (iii) increased risk of errors and difficulties
in reproducibility due to higher code complexity, while (iv) workflows are often not easily transferable to other
datasets, nor (v) scalable and transferable to changing computing resources.

scida aims to provide a solution to these problems by providing a simple interface to large datasets, while
hiding the complexity of the underlying data format and the parallelization of the computation.
Scida wraps around the dask library to provide a convenient way to provide a numpy/pandas-like interface to such data
with unit support via pint. Dask constructs task graphs for each operation, separating the definition of the computation from its execution.

At the time of writing, support is focused on astrophysical simulations and observations, but the package is
designed to be easily extensible to other scientific domains. Common analysis frameworks for astrophysical
simulations include python packages such as yt [@yt], pynbody [@pynbody], and swiftsimio [@Borrow2021]. These projects provide a limited to extensive
support of underlying data formats, providing specific common analysis routines for these astrophysical data sets and
include unit support.
Direct data access allowing full flexibility, however, is primarily made available through explicit loading of
numpy arrays into memory. This approach is not scalable to large data sets, and requires the user to explicitly
manage the data chunks in custom analysis routines.

Instead, scida provides an established interface via the dask library to handle large data sets in a scalable fashion,
allowing to leverage dask functionality and that of dask-based libraries such as dask-image [@dask-image] and datashader [@datashader].


# Target Audience

Scida aims to simplify access to large scientific data sets and lowering the burden adopting the scalable dask library.
As such, the scida package is targeted at researchers new to the field of large data analysis, but also at more
experienced researchers who want to make their workflow easier to read and scalable. Domain specific analysis
routines are primarily to be implemented on top of scida. Initial data support is currently focused on astrophysical
data sets, but scida aims to support other scientific domains as well, where similar solutions are missing.

# References
