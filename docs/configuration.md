# Configuration

## Main configuration file
The main configuration file is located at `~/.scida/config.yaml`. If this file does not exist, it is created with the
first use of scida. The file is using the YAML format.
The following options are available:

`copied_default`

:  If this option is set True, a warning is printed because the copied default config has not been adjusted by the user
   yet. Once you have done so, remove this line.

`cache_path`

: Sets the folder to use as a cache for scida. Recommended to be moved out of the home directory to a fast disk.

`datafolders`

: A list of folders to scan for data specifiers when using `scida.load("specifier")`.

`nthreads`

: scida itself might use multiple threads for some operations. This option sets the number of threads to use.
  This is independent of any dask threading. Default: 8

`missing_units`

: How to handle missing units. Can be "warn", "raise", or "ignore". "warn" will print a warning, "raise" will raise an
  exception, and "ignore" will silently continue without the right units. Default: "warn"
