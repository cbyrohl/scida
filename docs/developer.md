# Developer Guide

We welcome contributions to scida, such as bug reports, feature requests, and design proposals.
This page contains information on how to contribute to scida.


## Development environment

### Clone the repository

Make a fork of the [repository](https://github.com/cbyrohl/scida), then clone the repository to your local machine:

``` bash
git clone https://github.com/YOURUSERNAME/scida
cd scida
```

### Install

We use [poetry](https://python-poetry.org/) to manage dependencies and the development environment.
After installing poetry, you can install scida and its dependencies with

``` bash
poetry install
```

This will create a virtual environment and install scida and its dependencies,
including development dependencies.
All commands, such as `python` and `pytest` will be run in this environment
by prepending `poetry run ...` to the command.

While using poetry is recommended, you can also install scida with pip
in a virtual environment of your choice:

``` bash
python -m venv scida_venv
source scida_venv/bin/activate
pip install -e .
```

Note that in latter case, you will have to manage the dependencies yourself, including development dependencies.
If choosing this path, remove any `poetry run` prefixes from the commands below accordingly.

### Run tests

To run the tests, use

``` bash
poetry run pytest
```

Many tests require test data sets. These might not be available to you and lead to many tests being skipped.


## Contributing code

### Code Formatting

We use the [black](https://github.com/psf/black) code formatter to ensure a consistent code style.
This style is ensured by the [pre-commit hook config](https://github.com/cbyrohl/scida/blob/main/.pre-commit-config.yaml).
Make sure to have [pre-commit](https://pre-commit.com/) installed and run

``` bash
pre-commit install
```

in the repository to install the hook.

### Docstring style

We use [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) to format docstrings.
Please annotate all functions and classes with docstrings accordingly.

### Testing

We use [pytest](https://docs.pytest.org/en/stable/) for testing. Add new tests for added functionality
in a test file in the `tests` directory. Make sure to run the tests before submitting a pull request.
