# Frequently Asked Questions
## Extending existing datasets
> How do I add custom fields (that are not derived fields) to my existing dataset?

*Please also see how to add [derived fields](derived-fields.md).*

After loading some dataset, you can add custom fields to a given container, here called *PartType0* by simply assigning a dask array under the desired key to it.

Please note that all fields within a container are expected to have the same shape in their first axis.

``` py
from astrodask import load
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
