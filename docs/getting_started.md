# Getting started

## Installation

scida can be installed via [PyPI](https://pypi.org/). scida requires a Python version between 3.9 and 3.11.

!!! tip "Encapsulating packages"
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


!!! error inline end "Not working"

    This will only start working upon public release. For now use
    ``` bash
    python -m pip install 'scida @ git+https://github.com/cbyrohl/scida
    ```
    if you have been granted access to the code repository.


``` bash
pip install scida
```


Next, get started with the [tutorial](tutorial.md).
