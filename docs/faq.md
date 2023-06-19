# Frequently Asked Questions
## Extending existing datasets
> How do I add custom fields (that are not derived fields) to my existing dataset?

*Please also see how to add [derived fields](derived_fields.md).*

After loading some dataset, you can add custom fields to a given container, here called *PartType0* by simply assigning a dask array under the desired key to it.

Please note that all fields within a container are expected to have the same shape in their first axis.

``` py
from scida import load
import da.array as da
ds = load('simname')
array = da.zeros_like(ds.data["PartType0"]["Density"][:,0])
ds.data['PartType0']["zerofield"] = array
```

As we operate with dask, make sure to cast your array accordingly. For example, if your field is just a numpy array or a hdf5 memmap, you can use `da.from_array` to cast it to a dask array.
Alternatively, if you have another dataset loaded, you can assign fields from one to another:

``` py
ds2 = load('simname2')
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
