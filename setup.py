import setuptools

setuptools.setup(
    name='astrodask',
    version='0.0.1',
    description='Python tools for analyzing of AREPO simulations utilizing dask for scalability.',
    packages=['astrodask'],
    python_requires=">=3.7",
    install_requires=[
        'h5py',
        'numpy',
        'numba',
        'pandas',
        'astropy',
        'dask',
        'distributed',
        'zarr'
    ],
    tests_require=[
        'pytest',
        'ddt'
    ],
)
