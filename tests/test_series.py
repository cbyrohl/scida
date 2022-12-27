import numpy as np

from astrodask.series import ArepoSimulation
from tests.testdata_properties import require_testdata_path


@require_testdata_path("areposimulation")
def test_areposimulation_load(testdatapath):
    bs = ArepoSimulation(testdatapath)
    ds = bs.datasets[0]
    print(list(ds.data.keys()))
    print(bs.redshifts)
    redshift = 2.0
    ds = bs.get_dataset(redshift=redshift)
    print(redshift, ds.redshift)
    assert np.isclose(redshift, ds.redshift, rtol=1e-2)
    # assert testdatapath_areposimulation.file is not None
