# scida

[![pyversions](https://img.shields.io/pypi/pyversions/scida)](https://pypi.org/project/scida/)
![test status](https://github.com/cbyrohl/scida/actions/workflows/tests.yml/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06064/status.svg)](https://doi.org/10.21105/joss.06064)

scida is an out-of-the-box analysis tool for large scientific datasets. It primarily supports the astrophysics community, focusing on cosmological and galaxy formation simulations using particles or unstructured meshes, as well as large observational datasets.
This tool uses dask, allowing analysis to scale up from your personal computer to HPC resources and the cloud.

## Features

- Unified, high-level interface to load and analyze large datasets from a variety of sources.
- Parallel, task-based data processing with dask arrays.
- Physical unit support via pint.
- Easily extensible architecture.

## Requirements

- Python 3.9, 3.10, 3.11


## Documentation
The documentation can be found [here](https://cbyrohl.github.io/scida/).

## Install

```
pip install scida
```

## First Steps

After installing scida, follow the [tutorial](https://cbyrohl.github.io/scida/tutorial/).

## Citation

If you use scida in your research, please cite the following [paper](https://joss.theoj.org/papers/10.21105/joss.06064):

```text
`Byrohl et al., (2024). scida: scalable analysis for scientific big data. Journal of Open Source Software, 9(94), 6064, https://doi.org/10.21105/joss.06064`
```

with the following bibtex entry:

```text
@article{scida,
  title = {scida: scalable analysis for scientific big data},
  author = {Chris Byrohl and Dylan Nelson},
  doi = {10.21105/joss.06064},
  url = {https://doi.org/10.21105/joss.06064}, year = {2024},
  publisher = {The Open Journal}, volume = {9}, number = {94},
  pages = {6064},
  journal = {Journal of Open Source Software}
}
```

## Issues

If you encounter any problems,
please [file an issue](https://github.com/cbyrohl/scida/issues/new) along with a detailed description.

## License

Distributed under the terms of the [MIT license](LICENSE),
_scida_ is free and open source software.
