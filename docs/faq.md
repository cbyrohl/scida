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


## Misc
> How does load() determine the right type of dataset/series to load?

*load()* will step through all subclasses of *Series()* and *Dataset()* and call their *validate_path()* class method.
A list of candidate classes that return *True* upon this call is assembled. If more than one candidate exists,
the most specific candidate, i.e. the one furthest down the inheritance tree, is chosen.

The candidate can be overwritten when a YAML configuration specifies "dataset_type/series" and/or "dataset_type/dataset" keys to the respective class name.

In addition to this, different features, such as for datasets using Cartesian coordinates, are added
as so-called [Mixins](https://en.wikipedia.org/wiki/Mixin) to the dataset class.
