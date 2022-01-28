# We test on a snapshot from TNG50-4
from . import path,gpath

from ..interface import BaseSnapshot
import dask.array as da

def test_fields():
    snp = BaseSnapshot(path)
    gas_field_register = snp.data["PartType0"].register_field()

    @gas_field_register
    def testfield(data, **kwargs):
        return da.zeros(1)

    @gas_field_register
    def testfield2(data, **kwargs):
        return data["testfield"]+1

    @snp.register_field("PartType0", name="testfield3")
    def testfield_explicitname(data, **kwargs):
        return data["testfield"]+2

    # first-order field
    val0 = snp.data["PartType0"]["testfield"].compute()
    assert val0 == 0
    # second-order field
    val1 = snp.data["PartType0"]["testfield2"].compute()
    assert val1 == 1
    # second-order field with custom name
    val2 = snp.data["PartType0"]["testfield3"].compute()
    assert val2 == 2
