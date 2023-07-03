# Units

## Loading data with units

Loading data sets with 

``` py
from scida import load
ds = load("TNG50-4_snapshot")
```

will automatically attach units to the data. This can be deactivated by passing "units=False" to the load function.
By default, code units are used, alternatively, cgs conversions can be applied by passing "units='cgs'".

Units are introduced via the [pint](https://pint.readthedocs.io/en/stable/) package, see there for more details.


``` pycon
>>> gas["Coordinates"]
dask.array<mul, shape=(18540104, 3), dtype=float64, chunksize=(5592405, 3), chunktype=numpy.ndarray> <Unit('code_length')>
```

We can access the underlying dask array and the units separately:

``` pycon
>>> gas["Coordinates"].magnitude, gas["Coordinates"].units
(dask.array<mul, shape=(18540104, 3), dtype=float64, chunksize=(5592405, 3), chunktype=numpy.ndarray>,
 <Unit('code_length')>)
```

## Unit conversions

We can change units for evaluation as desired:

``` pycon
>>> coords = gas["Coordinates"]
>>> coords.to("cm") 
>>> # here the default system is cgs, thus we get the same result from
>>> coords.to_base_units()
dask.array<mul, shape=(18540104, 3), dtype=float64, chunksize=(5592405, 3), chunktype=numpy.ndarray> <Unit('centimeter')>
```

# The unit registry

The unit registry keeps all units. There is no global registry, but each dataset has its own registry as attribute ureg.
The use of a global registry (or lack thereof here) can lead to some confusion, please consult the pint documentation when in doubt.

``` pycon
>>> ureg = ds.ureg
>>> # get the unit meter
>>> ureg("m")
1 <Unit('meter')>
```


``` pycon
>>> # define an array with units meter (dask arrays analogously)
>>> import numpy as np
>>> np.arange(10) * ureg("m")
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) <Unit('meter')>
```

## Synthesize new dask arrays with units

``` pycon
>>> energy_restframe = (gas["Masses"]*ureg("c")**2).to("erg")  # E=mc^2
>>> energy_restframe
dask.array<mul, shape=(18540104,), dtype=float64, chunksize=(18540104,), chunktype=numpy.ndarray> <Unit('erg')>
```

## Custom units

``` pycon
>>> ureg.define("halfmeter = 0.5 * m")
>>> # first particle coordinates in halfmeters
>>> coords.to("halfmeter")[0].compute()
array([6.64064027e+23, 2.23858253e+24, 1.94176712e+24]) <Unit('halfmeter')>
```


