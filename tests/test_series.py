import os
import time

import numpy as np

from astrodask.series import ArepoSimulation
from tests.testdata_properties import require_testdata_path


@require_testdata_path("areposimulation")
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
    ds = bs2.get_dataset(0)
    assert ds.header is not None
