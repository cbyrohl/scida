# Frequently Asked Questions
## Common warnings/exceptions
> I get the warning "UserWarning: Passing an object to dask.array.from_array which is already a Dask collection. This can lead to unexpected behavior.". What does this mean?

In scida, this warning often occurs when passing a field with units to a dask.array function.
Consider explicitly removing the units (e.g. using "arr.magnitude" for the field/array "arr") before the dask.array function call.
Then attach the correct units afterwards again. This will be automatized in the future.

> "ValueError: Cannot operate with Quantity and Quantity of different registries."

Likely, you are trying to combine fields using different unit registries, e.g. when comparing fields from different datasets. This is not allowed with [pint](https://pint.readthedocs.io/en/stable).

If you can assure that the definitions of the units for a given field are the same between registries,
you could use the following workaround:

``` py hl_lines="4-7"
from scida import load
ds = load("./snapdir_030")
ds2 = load("./snapdir_030")
unitregistry = ds.ureg
field = ds2.data['PartType0']["Masses"]
units = unitregistry(str(field.units))
field = field.magnitude * units
ds.data['PartType0']["Masses2"] = field
```

> "AttributeError: Series do not have 'data' attribute. Load a dataset from series.get_dataset()."

You most likely invoked the [load](https://scida.io/api/base_api/#convenience-functions) function prior to accessing the data attribute. Depending on the path passed to the load function, it returns an instance of a DataSeries or Dataset. DataSeries are collections of Datasets, and do not have the ".data" attribute. You can either access a dataset of the series (e.g. "sim.get_dataset(0)" where 0 is the index of the dataset to load), or directly load the dataset by specifying the correct subdirectory. You can subsequently call the data attribute on the dataset.


## Extending existing datasets
> How do I add custom fields (that are not derived fields) to my existing dataset?

While [derived fields](derived_fields.md) are the preferred way to add fields, you can also add custom dask arrays to a dataset.

After loading some dataset, you can add custom fields to a given container, here called *PartType0* by simply assigning a dask array under the desired field name:

``` py
import dask.array as da
ds = load("./snapdir_030")
array = da.zeros_like(ds.data["PartType0"]["Density"])
unitregistry = ds.ureg
ds.data['PartType0']["zerofield"] = array * unitregistry("m")
```

!!! info

    Please note that all fields within a container are expected to have the same shape in their first axis. When attaching units, it is important to use the dataset's unit registry `ds.ureg`.

As scida expects dask arrays, make sure to cast your array accordingly. For example, if your field is just a numpy array or a hdf5 memmap, you can use `da.from_array` to cast it to a dask array.
Alternatively, if you have another dataset loaded, you can assign fields from one to another:

``` py
ds2 = load("./snapdir_030")
ds.data['PartType0']["NewDensity"] = ds2.data['PartType0']["Density"]
```

## What libraries is scida built on?

Scida strongly relies on the following libraries:
- [dask](https://dask.org/): For parallel computing
- [pint](https://pint.readthedocs.io/en/stable): For handling units

Data support is provided, among others, by:

- [h5py](https://www.h5py.org/): For reading HDF5 files
- [zarr](https://zarr.readthedocs.io/en/stable/): For reading Zarr files

Visualization in the examples is commonly done with:

- [matplotlib](https://matplotlib.org/): For plotting
- [bokeh](https://docs.bokeh.org/en/latest/index.html): For interactive plotting
- [holoviz](https://holoviz.org/): high-level interactive visualization

The original repository structure was inspired by the following templates:

- [wolt template](https://github.com/woltapp/wolt-python-package-cookiecutter)
- [hypermodern python template](https://github.com/cjolowicz/cookiecutter-hypermodern-python)

These lists are not exhaustive. Also see the [pyproject.toml](https://github.com/cbyrohl/scida/blob/main/pyproject.toml)
file for a list of dependencies.



## Misc
> How does load() determine the right type of dataset/series to load?

*load()* will step through all subclasses of *Series()* and *Dataset()* and call their *validate_path()* class method.
A list of candidate classes that return *True* upon this call is assembled. If more than one candidate exists,
the most specific candidate, i.e. the one furthest down the inheritance tree, is chosen.

The candidate can be overwritten when a YAML configuration specifies "dataset_type/series" and/or "dataset_type/dataset" keys to the respective class name.

In addition to this, different features, such as for datasets using Cartesian coordinates, are added
as so-called [Mixins](https://en.wikipedia.org/wiki/Mixin) to the dataset class.
