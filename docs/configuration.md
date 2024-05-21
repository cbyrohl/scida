# Configuration

## Main configuration file
The main configuration file is located at `~/.config/scida/config.yaml`. If this file does not exist, it is created with the
first use of scida. The file is using the YAML format.
The following options are available:

`copied_default`

:  If this option is set True, a warning is printed because the copied default config has not been adjusted by the user
   yet. Once you have done so, remove this line.

`cache_path`

: Sets the folder to use as a cache for scida. Recommended to be located on a fast disk.

`datafolders`

: A list of folders to scan for data specifiers when using `scida.load("specifier")`.

`nthreads`

: scida itself might use multiple threads for some operations. This option sets the number of threads to use.
  This is independent of any dask threading. Default: 8

`missing_units`

: How to handle missing units. Can be "warn", "raise", or "ignore". "warn" will print a warning, "raise" will raise an
  exception, and "ignore" will silently continue without the right units. Default: "warn"

`testdata_path`
: The base path to the test data sets defined in "tests/testdata.yaml".

## Simulation configuration
By default, scida will load supported [simulation configurations from the package](https://github.com/cbyrohl/scida/blob/main/src/scida/configfiles/simulations.yaml).
User configurations for simulations are loaded from `~/.config/scida/simulations.yaml`. This file is also in YAML format.

The configuration has to have the following structure:
```yaml
data:
  SIMNAME1:

  SIMNAME2:

```

Each simulation could look something like this:

```yaml
data:
  SIMNAME1:
    aliases:
      - SIMNAME
      - SMN1
    identifiers:
      Parameters:
        SimName: SIMNAME1
      Config:
        SavePath:
          content: /path/to/simname
          match: substr
    unitfile: units/simnameunits.yaml
    dataset_type:
      series: ArepoSimulation
      dataset: ArepoSnapshot
```

`aliases`

: A list of aliases for the simulation. These can be used to load the simulation with `scida.load("alias")`.

`identifiers`

: A dictionary of identifiers from the metadata of a given dataset to identify it as such.
  In above example "/Parameters" is the path to an attribute "SimName" in the HDF5/zarr metadata
  with the exact content as given. Multiple identifiers can be given, in which case all have to match.
  Partial matches of a given key-value key are possible by passing a dictionary {"content": "valuesubstr", match: substring}
  rather than a string.

`unitfile`

: The path to the unitfile relative to the user/repository simulation configuration. user configurations
  take precedence over the package configuration.

`dataset_type`

: Can explicitly fix the dataset/series type for a simulation.


## Unit files
Unit files are used to determine the units of datasets, particularly for datasets that do not have metadata
that can be used to infer units. Unit files are specified either explicitly via the `unitfile` option in `scida.load`
or implicitly via the simulation configuration, see above. Relative paths, such as `units/simnameunits.yaml` are
relative to the user/package simulation config folder. The former (`~/.config/scida/`) takes precedence.

A unit file could look like this:

```yaml
metadata_unitsystem: cgs
units:
  unit_length: 100.0 * km
  unit_mass: g
fields:
  _all:
    CounterID: none
    Coordinates: unit_length
  InternalArrays: none
  PartType0:
    SubPartType0:
      FurthestSubgroupDistance: unit_length
    NearestNeighborDistance: unit_length
    Energy: 10.0 * erg
```

`metadata_unitsystem`

: The unitsystem assumed when deducing units from metadata dimensions where available.
  Only cgs supported right now.

`units`

: unit definitions that are used in the following `fields` section. The units are defined as
  [pint](https://pint.readthedocs.io/en/stable/) expressions.

`fields`

: A dictionary of fields and their units. The fields are specified as a path to the field in the dataset.
  The special field `_all` can be used to set the default unit for all fields with a given name irrespective
  of the path of the field. Other than that, entries represent the fields or containers of fields. The special
  field `none` can be used to set the unit to None, i.e. no unit. This is differently handled than " "/"dimensionless" as
  the field will be treated as array rather than dimensionless [pint](https://pint.readthedocs.io/en/stable/) array.
