# Derived fields

Commonly during analysis, newly derived quantities/fields are to be synthesized from one or more snapshot fields into a new field. For example, while the temperature, pressure, or entropy of gas is not stored directly in the snapshots, they can be computed from fields which are present on disk.

There are two ways to create new derived fields. For quick analysis, we can simply leverage dask arrays themselves.


## Defining new quantities with dask arrays

``` py
from astrodask import load
ds = load("somedataset") # (1)!
gas = ds.data['gas']
kineticenergy = 0.5*gas['Masses']*gas['Velocities']**2
```

1.  In this example, we assume a dataset, such as the 'TNG50\_snapshot' test data set, that has its fields (*Masses*, *Velocities*) nested by particle type (*gas*)

In the example above, we define a new dask array called kineticenergy. Note that just like all other dask arrays and dataset fields, these fields are "virtual", i.e. only the graph of their construction is held in memory, which can be instantiated by applying the *.compute()* method.

We can also add this field from above example to the existing ones in the dataset.

``` py
gas['kineticenergy'] = kineticenergy
```


## Defining new quantities with field recipes

Working with complex datasets over a longer period, it is often useful to have a large range of fields available. The above approach with dask arrays suffers from some shortcomings. For example, in some cases the memory footprint and instantiation time for each field can add up to substantial loading times. Also, when defining fields with dask arrays, these fields need to be defined in order of their respective dependencies.

For this purpose, **field recipes** are available. An example of such recipe is given below.


``` py
import dask.array as da

from astrodask import load
ds = load("somedataset")

@snap.register_field("stars")  # (1)!
def VelMag(arrs, **kwargs):
    vel = arrs['Velocities']
    return np.sqrt( vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2 )
```

1.  Here, *stars* is the name of the **field container** the field should be added to. The field will now be available as ds\['stars'\]\['VelMag'\]

The field recipe is translated into a regular field, i.e. dask array, the first time it is queried for. Above example can be queried as:

``` py
ds['stars']['VelMag']
```

Practically working with these fields, there is no difference between derived and on-disk fields.
