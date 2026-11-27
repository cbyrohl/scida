import dask
import numpy as np
import pytest
from astropy.io import fits

from scida import load


@pytest.mark.integration
@pytest.mark.parametrize("nrows", [0, 5])
def test_load_fixed_length_fits_vector(tmp_path, nrows):
    path = tmp_path / "vectors.fits"
    vectors = np.arange(nrows * 3, dtype=np.float32).reshape(nrows, 3)
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="flux", format="3E", array=vectors),
            fits.Column(name="id", format="J", array=np.arange(nrows)),
        ]
    ).writeto(path)

    with dask.config.set({"array.chunk-size": "24B"}):
        dataset = load(path, units=False)
    assert dataset.data["flux"].shape == (nrows, 3)
    if nrows:
        assert dataset.data["flux"].chunks[0] == (2, 2, 1)
    np.testing.assert_array_equal(dataset.data["flux"].compute(), vectors)
    np.testing.assert_array_equal(dataset.data["id"].compute(), np.arange(nrows))


@pytest.mark.integration
def test_load_reports_unsupported_fits_column(tmp_path, caplog):
    path = tmp_path / "ragged.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ragged", format="PJ()", array=[[1], [2, 3]]),
            fits.Column(name="id", format="J", array=[10, 11]),
        ]
    ).writeto(path)

    dataset = load(path, units=False)

    assert "Skipping unsupported variable-length FITS column 'ragged'" in caplog.text
    assert "ragged" not in dataset.data
    np.testing.assert_array_equal(dataset.data["id"].compute(), [10, 11])
