# Getting started

## Installation

astrodask can be installed via [PyPI](https://pypi.org/). astrodask requires a Python version between 3.9 and 3.11.

!!! tip "Encapsulating packages"
    We recommend encapsulating your python environments. For example using [anaconda](https://www.anaconda.com/) or [virtualenv](https://virtualenv.pypa.io/en/latest/).

    If you use anaconda, we recommend running

    ```
    conda create -n astrodask python=3.9
    ```

    Activate the environment as needed (as for the following installation) as

    ``` bash
    conda activate astrodask
    ```

    If you are using jupyter/ipython, install and register the astrodask kernel via


    ``` bash
    conda install ipykernel
    python -m ipykernel install --user --name astrodask --display-name "astrodask"
    ```


!!! error inline end "Not working"

    This will only start working upon public release. For now use
    ``` bash
    python -m pip install 'astrodask @ git+https://github.com/cbyrohl/astrodask
    ```
    if you have been granted access to the code repository.


``` bash
pip install astrodask
```


Next, get started with the [tutorial](tutorial.md).
