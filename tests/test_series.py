import os
import time

import numpy as np
import psutil
import pytest
from IPython.core.formatters import HTMLFormatter

from scida import ArepoSimulation
from scida.interface import BaseDataset
from scida.series import delay_init
from tests.testdata_properties import require_testdata_path


@pytest.mark.external
@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_load(testdatapath):
    bs = ArepoSimulation(testdatapath)
    ds = bs.datasets[0]
    print(list(ds.data.keys()))
    redshift = 2.0
    ds = bs.get_dataset(redshift=redshift)
    print(ds)
    assert len(bs) == 5
    assert np.isclose(redshift, ds.header["Redshift"], rtol=1e-2)

    # some other keywords
    ds1 = bs.get_dataset(t=1.0)
    ds2 = bs.get_dataset(z=0.0)
    assert ds1.__repr__() == ds2.__repr__()

    ds1 = bs.get_dataset(index=1)
    ds2 = bs.get_dataset(snap=1)
    assert ds1.__repr__() == ds2.__repr__()

    # cannot specify multiple aliases for "index"
    with pytest.raises(ValueError):
        bs.get_dataset(index=1, snap=1)
    with pytest.raises(ValueError):
        bs.get_dataset(snapshot=1, snap=1)


@pytest.mark.external
@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_load_property_tolerance(testdatapath):
    bs = ArepoSimulation(testdatapath)
    bs.get_dataset(redshift=2.0)
    with pytest.raises(ValueError):
        # the next snapshot redshifts are 2.0 and 3.0, thus we should get an error
        # for the default tolerance.
        bs.get_dataset(redshift=2.5)


@pytest.mark.external
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


@pytest.mark.external
@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_delay_obj(testdatapath):
    # lazy loading
    cls = BaseDataset
    snp = delay_init(cls)(testdatapath)
    assert "Delay" in str(type(snp))

    # string representation should not trigger evaluation
    print(snp)  # triggers __repr__
    assert "Delay" in str(type(snp))

    # ipython display should also not trigger evaluation
    fmt = HTMLFormatter()
    print(fmt(snp))
    assert "Delay" in str(type(snp))

    # accessing attributes should trigger evaluation
    print(snp.data)
    assert "Delay" not in str(type(snp))


@pytest.mark.external
@require_testdata_path("areposimulation", only=["TNGvariation_simulation", "TNG50-4"])
def test_areposimulation_lazy_message(cachedir, testdatapath):
    tstart = time.process_time()
    bs = ArepoSimulation(testdatapath, lazy=True)
    dt0 = time.process_time() - tstart
    print(bs.datasets[0])
    assert type(bs.datasets[0]).__name__ == "Delay"
    print(dt0)
    assert dt0 < 4.0  # should be fast


@pytest.mark.external
@require_testdata_path("areposimulation", only=["TNGvariation_simulation", "TNG50-4"])
def test_areposimulation_lazy(cachedir, testdatapath):
    tstart = time.process_time()
    bs = ArepoSimulation(testdatapath, lazy=True)
    dt0 = time.process_time() - tstart
    print(bs.datasets[0])
    assert type(bs.datasets[0]).__name__ == "Delay"
    print(dt0)
    assert dt0 < 4.0  # should be fast


@pytest.mark.external
@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_lazy2(cachedir, testdatapath):
    # first load of simulation in lazy mode
    # tstart = time.process_time()
    bs1 = ArepoSimulation(testdatapath, lazy=True)
    # dt0 = time.process_time() - tstart
    assert type(bs1.datasets[0]).__name__ == "Delay"

    # second load of simulation in lazy mode
    # tstart = time.process_time()
    # bs2 = ArepoSimulation(testdatapath, lazy=False)
    # dt1 = time.process_time() - tstart
    # TODO: Figure out why this one is not "Delay".
    # assert type(bs2.datasets[0]).__name__ != "Delay"
    # TODO: Recheck. Now since we are always lazily evaluating and using metadata, this test does not make sense...
    # assert 10 * dt0 < dt1

    assert type(bs1.datasets[2]).__name__ == "Delay"
    assert bs1.datasets[2].data is not None
    assert type(bs1.datasets[2]).__name__ != "Delay"

    assert type(bs1.datasets[1]).__name__ == "Delay"
    bs1.datasets[1].evaluate_lazy()
    assert type(bs1.datasets[1]).__name__ != "Delay"


@pytest.mark.external
@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_caching(cachedir, testdatapath):
    # first call to cache datasets
    # tstart = time.process_time()
    bs = ArepoSimulation(testdatapath)
    # dt0 = time.process_time() - tstart
    if os.path.exists(bs._metadatafile):
        os.remove(bs._metadatafile)  # remove series metadata cache file

    # second call to cache dataseries, and only series
    # tstart = time.process_time()
    bs1 = ArepoSimulation(testdatapath)
    # dt1 = time.process_time() - tstart

    # third to benchmark use with dataseries cache
    # tstart = time.process_time()
    bs2 = ArepoSimulation(testdatapath)
    # dt2 = time.process_time() - tstart

    # simple check that cache match at first level
    md1 = bs1.metadata
    md2 = bs2.metadata
    assert set(md1) == set(md2) != set()

    # cache vs no cache
    # TODO: Recheck. Now since we are always lazily evaluating and using metadata, this test does not make sense...
    # assert 5 * dt2 < dt1 < dt0, "dataseries caching does not speed up as intended"

    # try to use a dataset from cached version
    redshift = 2.0
    print("md", bs1.metadata)
    ds1 = bs1.get_dataset(redshift=redshift)
    ds2 = bs2.get_dataset(redshift=redshift)
    assert np.isclose(ds1.metadata["redshift"], redshift, rtol=1e-2)
    assert np.isclose(ds2.metadata["redshift"], redshift, rtol=1e-2)

    assert ds2.header is not None


@pytest.mark.external
@require_testdata_path("areposimulation", only=["TNGvariation_simulation"])
def test_areposimulation_mixins(cachedir, testdatapath):
    bs = ArepoSimulation(testdatapath)
    ds = bs.get_dataset(0)
    assert ds.header is not None  # need some call to instantiate Delay obj
    assert "CosmologyMixin" in type(ds).__name__
