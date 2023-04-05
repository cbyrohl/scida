import os
import time

import numpy as np
import psutil

from astrodask.series import ArepoSimulation
from tests.testdata_properties import require_testdata_path


@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_load(testdatapath):
    bs = ArepoSimulation(testdatapath)
    ds = bs.datasets[0]
    print(list(ds.data.keys()))
    print(bs.metadata)
    redshift = 2.0
    ds = bs.get_dataset(redshift=redshift)
    print(ds)
    print(redshift, ds.redshift)
    assert np.isclose(redshift, ds.redshift, rtol=1e-2)
    # assert testdatapath_areposimulation.file is not None


@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_asynccaching(cachedir, testdatapath):
    mstart = psutil.Process().memory_info().rss
    print(mstart)
    print(psutil.Process().memory_full_info())
    # d = np.zeros((1024,1024,1024))
    tstart = time.process_time()
    bs = ArepoSimulation(testdatapath)  # , lazy=True, async_caching=True)
    dt0 = time.process_time() - tstart
    print(dt0)
    m0 = psutil.Process().memory_info().rss - mstart
    print(psutil.Process().memory_info().rss)
    print(psutil.Process().memory_full_info())
    print(m0 / (1024.0**2))
    print(bs)

    # assert type(bs.datasets[0]).__name__ == "Delay"


@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_lazy(cachedir, testdatapath):
    tstart = time.process_time()
    bs1 = ArepoSimulation(testdatapath, lazy=True)
    dt0 = time.process_time() - tstart
    assert type(bs1.datasets[0]).__name__ == "Delay"

    tstart = time.process_time()
    bs2 = ArepoSimulation(testdatapath, lazy=False)
    dt1 = time.process_time() - tstart
    assert type(bs2.datasets[0]).__name__ != "Delay"
    assert 10 * dt0 < dt1

    assert type(bs1.datasets[2]).__name__ == "Delay"
    bs1.datasets[2].data
    assert type(bs1.datasets[2]).__name__ != "Delay"

    assert type(bs1.datasets[1]).__name__ == "Delay"
    bs1.datasets[1].evaluate_lazy()
    assert type(bs1.datasets[1]).__name__ != "Delay"


@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_caching(cachedir, testdatapath):
    # first call to cache datasets
    tstart = time.process_time()
    bs = ArepoSimulation(testdatapath)
    dt0 = time.process_time() - tstart
    os.remove(bs._metadatafile)  # remove series metadata cache file

    # second call to cache dataseries, and only series
    tstart = time.process_time()
    bs1 = ArepoSimulation(testdatapath)
    dt1 = time.process_time() - tstart

    # third to benchmark use with dataseries cache
    tstart = time.process_time()
    bs2 = ArepoSimulation(testdatapath)
    dt2 = time.process_time() - tstart

    # simple check that cache match at first level
    md1 = bs1.metadata
    md2 = bs2.metadata
    assert set(md1) == set(md2)

    # cache vs no cache
    assert 5 * dt2 < dt1 < dt0, "dataseries caching does not speed up as intended"

    # try to use a dataset from cached version
    redshift = 2.0
    ds1 = bs1.get_dataset(redshift=redshift)
    ds2 = bs2.get_dataset(redshift=redshift)
    assert np.isclose(ds1.metadata["redshift"], redshift, rtol=1e-2)
    assert np.isclose(ds2.metadata["redshift"], redshift, rtol=1e-2)

    assert ds2.header is not None
