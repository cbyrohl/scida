# astrodask

![buildstatus](https://github.com/cbyrohl/astrodask/actions/workflows/python-package-conda.yml/badge.svg)

astrodask is an out-of-the-box analysis tool for large astrophysical datasets, particularly cosmological and galaxy formation simulations using particles or unstructured meshes.
This tool uses dask, allowing analysis to scale up from your personal computer to HPC resources and the cloud.

## Features

- Task-based data processing with dask arrays.
- Unit support via pint

## Requirements

- Python >= 3.9


## Documentation
The documentation can be found [here](https://cbyrohl.github.io/astrodask/)

## Install

You can install _astrodask_ via [pip](TODO) from [PyPI](https://pypi.org/):

```console
$ pip install astrodask
```

or do a git clone of this repository:

```
git clone https://github.com/cbyrohl/astrodask
cd astrodask
pip install .
```


## First Steps
Start up a Jupyter notebook server and begin with the 'Getting Started' section of the documentation. You can also find the underlying notebook [here](docs/notebooks/gettingstarted.ipynb).

## License

Distributed under the terms of the [MIT license](LICENSE),
_astrodask_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue](https://github.com/cbyrohl/astrodask/issues/new) along with a detailed description.
