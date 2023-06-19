# scida

![buildstatus](https://github.com/cbyrohl/scida/actions/workflows/python-package-conda.yml/badge.svg)

scida is an out-of-the-box analysis tool for large scientific datasets. It primarily supports the astrophysics community, focusing on cosmological and galaxy formation simulations using particles or unstructured meshes, as well as large observational datasets.
This tool uses dask, allowing analysis to scale up from your personal computer to HPC resources and the cloud.

## Features

- Unified, high-level interface to load and analyze large datasets from a variety of sources.
- Parallel, task-based data processing with dask arrays.
- Physical unit support via pint.
- Easily extensible architecture.

## Requirements

- Python >= 3.9


## Documentation
The documentation can be found [here](https://cbyrohl.github.io/scida/)

## Install

You can install _scida_ via [pip](TODO) from [PyPI](https://pypi.org/):

```console
$ pip install scida
```

or do a git clone of this repository:

```
git clone https://github.com/cbyrohl/scida
cd scida
pip install .
```


## First Steps
Start up a Jupyter notebook server and begin with the 'Getting Started' section of the documentation. You can also find the underlying notebook [here](docs/notebooks/gettingstarted.ipynb).

## License

Distributed under the terms of the [MIT license](LICENSE),
_scida_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue](https://github.com/cbyrohl/scida/issues/new) along with a detailed description.
