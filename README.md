# scida

![test status](https://github.com/cbyrohl/scida/actions/workflows/tests.yml/badge.svg)

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
The documentation can be found [here](https://cbyrohl.github.io/scida/).

## Install

```
pip install scida
```

## First Steps
After installing scida, follow the [tutorial](https://cbyrohl.github.io/scida/tutorial/).

## License

Distributed under the terms of the [MIT license](LICENSE),
_scida_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue](https://github.com/cbyrohl/scida/issues/new) along with a detailed description.

## Acknowledgements

The project structure was adapted from [Wolt](https://github.com/woltapp/wolt-python-package-cookiecutter) and [Hypermodern Python](https://github.com/cjolowicz/cookiecutter-hypermodern-python) cookiecutter templates.
