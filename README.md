![buildstatus](https://github.com/cbyrohl/darepo/actions/workflows/python-package-conda.yml/badge.svg)

# astrodask
Analysis tool for astrophysical data, particularly galaxy formation and cosmological simulations run with AREPO. This tool tries to leverage the dask package for calculation of relevant properties. astrophysical datasets + dask = astrodask.

# Documentation
The public documentation is hosted [here](https://astrodask.cbyrohl.de/). (The latest build of the documentation can be found [here](https://byrohlc.pages.mpcdf.de/darepo/).)

# Install
Do a git clone of this repository. Then do

```
pip install -e .
```

in the cloned repository.

It is recommended to start from a clean environment. Using the [anaconda](https://www.anaconda.com/products/individual), create an environment from the provided specifications in the cloned repository, e.g.

```
conda env create -n astrodask -f ./environment.yml
pip install -e .
```

# First Steps
Start up a Jupyter notebook server and begin with the 'Getting Started' section of the documentation. You can also find the underlying notebook [here](docs/source/gettingstarted.ipynb).

