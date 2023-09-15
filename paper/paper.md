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
for scalable, parallel and/or out-of-core computation. Unit support is provided by the pint package. Primary data
formats are many-dimensional point clouds. Current file formats include zarr, multi-file HDF5, and FITS.

Data access is provided in a hierarchical dictionary-like data attribute for datasets initialized via the load() function.
See example below.
From there, any available dask (array) operation can be performed for the analysis.

```python
import dask.array as da
from scida import load

ds = load("TNG50")
pos = ds.data["PartType0"]["Coordinates"]
res = da.histogram2d(pos[:, 0], pos[:, 1], bins=100)
hist = res[0].compute()
```

# Statement of need
Today, scientific datasets often span terabytes in size. Manual handling of such data on a chunk-by-chunk basis
is not only tedious, but also error-prone. Scida wraps around the dask library to provide a convenient way to
provide a numpy/pandas like interface to such data with optional unit support via pint. Dask constructs task graphs
for each operation, separating the definition of the computation from its execution.

This abstraction allows (i) processing of data that is too large to fit into memory without explicit chunking,
(ii) parallel processing with multiple threads, processes and/or nodes on an HPC cluster or the cloud, (iii) definition of
complex workflows only to be executed when needed, and (iv) consistency of units throughout the computation.
