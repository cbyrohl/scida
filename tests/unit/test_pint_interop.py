import pint
import pytest
import numpy as np
import dask.array as da
from packaging import version
from pint import UnitRegistry

from scida.interfaces.mixins.units import update_unitregistry


@pytest.mark.unit
def test_update_unitregistry():
    ureg = UnitRegistry()
    update_unitregistry("units/general.yaml", ureg)
    ureg.define("h0 = 0.6774")
    ureg.define("code_length = 1.0 * kpc / h0")
    assert np.isclose((1.0 * ureg.Msun).to(ureg.g).magnitude, 1.98847e33)
    assert np.isclose(
        (1.0 * ureg.code_length).to(ureg.kpc).magnitude, 1.0 / 0.6774
    )


@pytest.mark.unit
def test_misc():
    ureg = UnitRegistry()
    print(ureg("dimensionless"))
    ureg.define("unknown = 1.0")
    a = 1.0 * ureg("unknown")
    print(a.dimensionality)
    print(ureg("unknown"))


@pytest.mark.unit
@pytest.mark.skipif(
    version.parse(pint.__version__) <= version.parse("0.20.1"),
    reason="pint bug, fixed in development, see https://github.com/hgrecco/pint/pull/1722",
)
def test_dask_pint1():
    ureg = pint.UnitRegistry()
    ureg.default_system = "cgs"
    ureg.define("halfmeter = 0.5 * meter")
    arr = da.ones(1) * ureg("halfmeter")
    print(arr[0].compute().to_base_units())


@pytest.mark.unit
def test_dask_pint2():
    ureg = pint.UnitRegistry()
    ureg.default_system = "cgs"
    ureg.define("halfmeter = 0.5 * meter")
    arr = da.ones(1) * ureg("halfmeter")
    print(arr[0].compute().magnitude)
    print(arr[0].to_base_units().compute())
