# Derived fields

!!! info

    If you want to run the code below, consider downloading the [demo data](supported_data.md#demo-data) or use the [TNGLab](supported_data.md#tnglab) online.

Commonly during analysis, newly derived quantities/fields are to be synthesized from one or more snapshot fields into a new field. For example, while the temperature, pressure, or entropy of gas is not stored directly in the snapshots, they can be computed from fields which are present on disk.

There are two ways to create new derived fields. For quick analysis, we can simply leverage dask arrays themselves.


## Defining new quantities with dask arrays

``` py
from scida import load
ds = load("./snapdir_030") # (1)!
gas = ds.data['gas']
kineticenergy = 0.5*gas['Masses']*(gas['Velocities']**2).sum(axis=1)
```

1. In this example, we assume a dataset, such as the [demo data set](supported_data.md#demo-data), that has its fields (*Masses*, *Velocities*) nested by particle type (*gas*)

In the example above, we define a new dask array called "kineticenergy". Note that just like all other dask arrays and dataset fields, these fields are "virtual", i.e. only the graph of their construction is held in memory, which can be instantiated by applying the *.compute()* method.

We can also add this field from above example to the existing ones in the dataset.

``` py
gas['kineticenergy'] = kineticenergy
```


## Defining new quantities with field recipes

Working with complex datasets over a longer period, it is often useful to have a large range of fields available. The above approach with dask arrays suffers from some shortcomings. For example, in some cases the memory footprint and instantiation time for each field can add up to substantial loading times. Also, when defining fields with dask arrays, these fields need to be defined in order of their respective dependencies.

For this purpose, **field recipes** are available. An example of such recipe is given below.


``` py
import numpy as np

from scida import load
ds = load("./snapdir_030")

@ds.register_field("stars")  # (1)!
def VelMag(arrs, **kwargs):
    import dask.array as da
    vel = arrs['Velocities']
    v, u = vel.magnitude, vel.units
    return da.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2) * u
```

1.  Here, *stars* is the name of the **field container** the field should be added to. The field will now be available as ds\['stars'\]\['VelMag'\]

The field recipe is translated into a regular field, i.e. dask array, the first time it is queried for. Above example can be queried as:

``` py
ds['stars']['VelMag']
```

Practically working with these fields, there is no difference between derived and on-disk fields.


## Adding multiple fields

It can be useful to write (a) dedicated field definition file(s). First, initialize a FieldContainer

``` py
from scida.fields import FieldContainer
groupnames = ["PartType0", "Subhalo"]  # (1)!
fielddefs = FieldContainer(containers=groupnames)

@fielddefs.register_field("PartType0") # (2)!
def Volume(arrs, **kwargs):
    return arrs["Masses"]/arrs["Density"]

@fielddefs.register_field("all") # (3)!
def GroupDistance3D(arrs, snap=None):
    """Returns distance to hosting group center. Returns rubbish if not actually associated with a group."""
    import dask.array as da
    boxsize = snap.header["BoxSize"]
    pos_part = arrs["Coordinates"]
    groupid = arrs["GroupID"]
    if hasattr(groupid, "magnitude"):
        groupid = groupid.magnitude
        boxsize *= snap.ureg("code_length")
    pos_cat = snap.data["Group"]["GroupPos"][groupid]
    dist3 = (pos_part-pos_cat)
    dist3, u = dist3.magnitude, dist3.units
    dist3 = da.where(dist3>boxsize/2.0, boxsize-dist3, dist3)
    dist3 = da.where(dist3<=-boxsize/2.0, boxsize+dist3, dist3) # PBC
    return dist3 * u

@fielddefs.register_field("all")
def GroupDistance(arrs, snap=None):
    import dask.array as da
    dist3 = arrs["GroupDistance3D"]
    dist3, u = dist3.magnitude, dist3.units
    dist = da.sqrt((dist3**2).sum(axis=1))
    dist = da.where(arrs["GroupID"]==-1, np.nan, dist) # set unbound gas to nan
    return dist * u
```

1. We define a list of field containers that we want to add particles to.
2. Specify the container we want to have the field added to.
3. Using the "all" identifier, we can also attempt to add this field to all containers we have specified.

Finally, we just need to import the *fielddefs* object (if we have defined it in another file) and merge them with a dataset that we loaded:

``` py
ds = load("./snapdir_030")
ds.data.merge(fielddefs)
```

In above example, we now have the following fields available:

``` py
gas = ds.data["PartType0"]
print(gas["Volume"])
print(gas["GroupDistance"])
```
