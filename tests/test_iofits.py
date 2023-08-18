import dask.array as da

from scida import load
from scida.io.fits import fitsrecords_to_daskarrays
from tests.testdata_properties import require_testdata_path


@require_testdata_path("fits", only=["SDSS_DR16_fits"])
def test_fitsread(testdatapath):
    path = testdatapath

    from astropy.io import fits

    ext = 1
    with fits.open(path, memmap=True, mode="denywrite") as hdulist:
        arr = hdulist[ext].data
    darrs = fitsrecords_to_daskarrays(arr)
    assert len(darrs) > 0
    for k, darr in darrs.items():
        assert isinstance(darr, da.Array)


@require_testdata_path("fits", only=["SDSS_DR16_fits"])
def test_fitsdataset(testdatapath):
    path = testdatapath

    # from scida.io import load as ioload
    # a,b,c,d = ioload(path)
    ds = load(path)
    print(ds)
