# Getting started

## Installation

scida can be installed via [PyPI](https://pypi.org/). scida requires a Python version 3.9, 3.10, 3.11 or 3.12.

??? tip "Encapsulating packages"
    We recommend encapsulating your python environments. For example using [anaconda](https://www.anaconda.com/) or [virtualenv](https://virtualenv.pypa.io/en/latest/).

    If you use anaconda, we recommend running

    ```
    conda create -n scida python=3.9
    ```

    Activate the environment as needed (as for the following installation) as

    ``` bash
    conda activate scida
    ```

    If you are using jupyter/ipython, install and register the scida kernel via


    ``` bash
    conda install ipykernel
    python -m ipykernel install --user --name scida --display-name "scida"
    ```

    Now you can install scida as described below and use it in jupyter notebooks with the given kernel.

``` bash
pip install scida
```

## Next steps
Next, get started with the [tutorial](tutorial/index.md) for either simulations or observations:

{!tutorial/index.md!}
