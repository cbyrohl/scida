import setuptools

setuptools.setup(
    name='darepo',
    version='0.0.1',
    description='Python tools for analyzing of AREPO simulations utilizing dask for scalability.',
    packages=['darepo'],
    python_requires=">=3.7",
    install_requires=[
        'h5py',
        'numpy',
        'numba',
        'pandas',
        'astropy',
        'dask',
        'distributed'
    ],
    tests_require=[
        'pytest',
        'ddt'
    ],
)
