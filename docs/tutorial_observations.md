# Tutorial (observational data set)

This package is designed to aid in the efficient analysis of large datasets, such as GAIA DR3.

!!! note "Tutorial dataset"
    In the following, we will subset from the [GAIA data release 3](https://www.cosmos.esa.int/web/gaia/dr3). The reduced dataset contains 100000 randomly selected entries only.
    Check [Supported Datasets](supported_data.md) for an incomplete list of supported datasets
    and requirements for support of new datasets.
    A tutorial for a cosmological simulation can be found [here](tutorial_simulations.md).


It uses the [dask](https://dask.org/) library to perform computations, which has several key advantages:

1. very large datasets which cannot normally fit into memory can be analyzed,
2. calculations can be automatically distributed onto parallel 'workers', across one or more nodes, to speed them up,
3. we can create abstract graphs ("recipes", such as for derived quantities) and only evaluate on actual demand.

## Loading an individual dataset

Here, we choose the [GAIA data release 3](https://www.cosmos.esa.int/web/gaia/dr3) as an example.
The dataset is obtained in HDF5 format as used at ITA Heidelberg. We intentionally select a small subset of the data to work with.
Choosing a subset means that the data size in the snapshot is small and easy to work with.
We demonstrate how to work with larger data sets at a later stage.

First, we load the dataset using the convenience function `load()` that will determine the appropriate dataset class for us:


```pycon title="Loading a dataset"
>>> from astrodask import load
>>> ds = load("TNG50-4_snapshot", units=True) #(1)!
>>> ds.info() #(2)!
class: DatasetWithUnitMixin
source: /home/cbyrohl/data/testdata-astrodask/gaia_dr3_subset100000.hdf5
=== Unit-aware Dataset ===
==========================
=== data ===
+ root (fields: 27, entries: 100000)
============
```

1. The `units=True` argument will attach code units to all fields (default). Alternative choices are *False* to go without units and *cgs* for cgs units.
   The current default is *False*, which will change to *True* in the near future.
2. Call to receive some information about the loaded dataset.

The dataset is now loaded, and we can inspect its contents, specifically its container and fields loaded.
We can access the data in the dataset by using the `data` attribute, which is a dictionary of containers and fields.

We have a total of 27 fields available, which are:

```pycon title="Available fields"
>>> ds.data.keys()
['pmdec',
 'distance_gspphot',
...
 'distance_gspphot_upper',
 'pmra_error']
```

Let's take a look at some field in this container:

```pycon title="Inspecting a field"
>>> ds.data["dec"]
dask.array<mul, shape=(100000,), dtype=float64, chunksize=(100000,), chunktype=numpy.ndarray> <Unit('degree')>
```

The field is a dask array, which is a lazy array that will only be evaluated when needed.
How these lazy arrays and their units work and are to be used will be explored in the next section.

## Dask arrays and units

### Dask arrays
Dask arrays are virtual entities that are only evaluated when needed.
If you are unfamiliar with dask arrays, consider taking a look at this [3-minute introduction](https://docs.dask.org/en/stable/array.html).

They are **not** numpy arrays, but they can be converted to them, and have most of their functionality.
Within dask, an internal task graph is created that holds the recipes how to construct the array from the underlying data.

In general, fields can be also be stored in flat or more nested structures, depending on the dataset.

We can trigger the evaluation of the dask array by calling `compute()` on it:

```pycon title="Evaluating a dask array"
>>> ds.data["dec"].compute()
array([ 0.86655069,  1.15477218,  2.14207063, ..., -1.5291509 ,
       -1.30061261, -0.88984633]) <Unit('degree')>
```

**However**, directly evaluating dask arrays is **strongly discouraged** for large datasets, as it will load the entire dataset into memory.
Instead, we will reduce the datasize by running desired analysis/reduction within dask before calling *compute()*,
which we present in the next section.

### Units

If passing `units=True` to `load()`, the dataset will be loaded with code units attached to all fields.
These units are attached to each field / dask array. Units are provided via the pint package.
See the [pint documentation](https://pint.readthedocs.io/en/stable/) for more information. Also check out the
[units cookbook](notebooks/cookbook/units.ipynb) for more examples.

In short, each field, that is represented by a modified dask array, has a magnitude (the dask array without any units attached) and a unit.
These can be accessed via the `magnitude` and `units` attributes, respectively.

```pycon  title="Accessing the magnitude and units of a field"
>>> ds.data["dec"].magnitude.compute(), ds.data["dec"].units
(dask.array<mul, shape=(100000,), dtype=float64, chunksize=(100000,), chunktype=numpy.ndarray>,
 <Unit('degree')>)
```

When defining derived fields from dask arrays, the correct units are automatically propagated to the new field,
and dimensionality checks are performed. Importantly, the unit calculation is done immediately, thus allowing
to directly see the resulting units and any dimensionality mismatches.


## Analyzing snapshot data
### Computing a simple statistic on (all) particles

The fields in our snapshot object behave similar to actual numpy arrays.

As a first simple example, let's calculate the total mass of gas cells in the entire simulation. Just as in numpy we can write

```pycon title="Calculating the total mass of gas cells"
>>> masses = ds.data["PartType0"]["Masses"]
>>> task = masses.sum()
>>> task
'dask.array<sum-aggregate, shape=(), dtype=float32, chunksize=(), chunktype=numpy.ndarray> code_mass'
```

Note that all objects remain 'virtual': they are not calculated or loaded from disk,
but are merely the required instructions, encoded into tasks.

TODO: remainder
