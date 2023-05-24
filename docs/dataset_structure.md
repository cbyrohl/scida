# Dataset structure

Here, we discuss the requirements for easy extension/support of new datasets.

## Supported file formats

Currently, input files need to have one of the following formats:

* [hdf5](https://www.hdfgroup.org/solutions/hdf5/)
* multi-file hdf5
* [zarr](https://zarr.readthedocs.io/en/stable/)

## Supported file structures

Just like this package, above file formats use a hierarchical structure to store data with three fundamental objects:

* **Groups** are containers for other groups or datasets.
* **Datasets** are multidimensional arrays of a homogeneous type, usually bundled into some *Group*.
* **Attributes** provide various metadata.

## Supported data structures

At this point, we only support unstructured datasets, i.e. datasets that do not depend on the memory layout for their
interpretation. For example, this implies that simulation codes utilizing uniform or adaptive grids are not supported.


## Examples of supported simulation codes

We explicitly support simulations run with the following codes:

* [Gadget](https://wwwmpa.mpa-garching.mpg.de/gadget4/)
* [Gizmo](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html)
* [Arepo](https://arepo-code.org/)
* [Swift](https://swift.strw.leidenuniv.nl/)
