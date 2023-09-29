---
title: 'scida: scalable analysis for scientific big data'
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

*scida* is a Python package for reading and analyzing large scientific data sets.
It uses the dask library for scalable, parallel and out-of-core computation [@Dask].
Data access is provided with a hierarchical dictionary-like data structure with a simple load() function.
Arbitrary custom analysis, as well as all available dask (array) operations, can be performed.
Computation requests from a user session are collected in a task graph.
The subsequent computation is executed only upon request, on a target resource (e.g. a HPC cluster, see Figure \ref{fig:sketch}).

![Schematic of the workflow. A dataset initialized with scida.load() holds virtual references to
 the underlying data. Operations are collected into a task graph via the dask library. Only upon issuing a compute()
command, the computation takes place on a target resource, that can differ from the user session.
Results are generally much smaller than the original data, and are sent back to the user session for further
analysis/plotting. \label{fig:sketch}](sketch.pdf)


# Features

Scida begins by providing a clean, abstract, dictionary-like interface to the underlying data, regardless of its
file format or structure on disk. Physical units are automatically inferred, when possible, and attached
to the dataset. Symbolic and automatic unit conversion is provided by the pint package [@Pint]. This metadata can
also be specified by the user via customizable configuration files.

Scida attempts to automatically determine the type of dataset loaded. At the file level, it currently supports
zarr, single/multi-file HDF5 and FITS files. The analysis functionality then available depends on the dataset type.
For example:

* Datasets which pair scattered pointsets with cataloged results of clustering algorithms have a
natural association between such groups and their member points. Cosmological simulations in astrophysics provide
one such motivating example, where halos and galaxies are identified in particle data. Scida then supports
broadcasting calculations over all groups.
* Datasets containing spatial coordinates in one, two, or three dimensions can be queried for spatial subsets.

The dataset-dependent functionality of scida is handled with a flexible and extensible mixin architecture.
Furthermore, scida can handle series of datasets, e.g. for time evolution or parameter variations.

# Statement of need

Today, scientific "big data" is petabyte-scale. This makes traditional analysis by a researcher on their personal
computer prohibitive. Writing analysis code which is distributed or out-of-core is complex, and is not the main
focus of scientists. The need for significant data management creates (i) a barrier for new researchers,
(ii) a substantial time commitment apart from the science itself, and (iii) an increased risk of errors and difficulties
in reproducibility due to higher code complexity, while (iv) workflows are often not easily transferable to other
datasets, nor (v) scalable and transferable to changing computing resources.

scida solves these problems by providing a simple interface to large datasets, hiding the complexity of the underlying data format
and file structures, and transparently handling the parallelization of analysis computations. This is handled by the dask
library, which naturally separates the definition of a given computation from its execution.

Initial support in scida is focused on astrophysical simulations and observations, but the package is
designed to be easily extensible to other scientific domains. Existing analysis frameworks for astrophysical
simulations include python packages such as yt [@yt], pynbody [@pynbody], tenet [@tenet], and swiftsimio [@Borrow2021].
None utilize the graph-based distributed analysis framework of dask. Often, existing analysis packages rely on the
explicit loading of entire datasets into main memory. However, this approach is not transparently scalable to large data sets,
and requires the user to explicitly manage the data chunks in custom analysis routines.

By providing a flexible interface to the dask library to handle large data sets in a scalable fashion,
users can also leverage dask functionality and dask-based libraries such as dask-image [@dask-image] and datashader [@datashader].


# Target Audience

Scida aims to simplify access to large scientific data sets. It lowers the barrier of entry for researchers to ask complex questions of big data.
As such, the scida package is targeted at researchers with large data analysis problems. Its clean interface is appropriate for users
with no prior experience with big data or distributed data analysis, as well as those who specifically want to leverage dask to make their
workflows easier to read and scalable.

Domain specific analysis routines can be implemented on top of scida. Initial "out of the box" data support is currently focused on astrophysical
data sets, but scida aims to support other scientific domains as well, where similar solutions are missing.

# References
