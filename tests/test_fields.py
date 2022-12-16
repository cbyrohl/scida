# We test on a snapshot from TNG50-4
import pytest

import dask.array as da


def test_fields(snp):
    gas_field_register = snp.data["PartType0"].register_field()

    @gas_field_register
    def testfield(data, **kwargs):
        return da.zeros(1)

    @gas_field_register
    def testfield2(data, **kwargs):
        return data["testfield"] + 1

    @snp.register_field("PartType0", name="testfield3")
    def testfield_explicitname(data, **kwargs):
        return data["testfield"] + 2

    # first-order field
    val0 = snp.data["PartType0"]["testfield"].compute()
    assert val0 == 0
    # second-order field
    val1 = snp.data["PartType0"]["testfield2"].compute()
    assert val1 == 1
    # second-order field with custom name
    val2 = snp.data["PartType0"]["testfield3"].compute()
    assert val2 == 2

    # test that we are passed the snapshot instance
    @snp.register_field("PartType0")
    def access_snapshot(data, **kwargs):
        return kwargs["snap"]

    val = snp.data["PartType0"]["access_snapshot"]
    assert val is not None

    @snp.register_field("PartType0")
    def access_snapshot2(data, snap=None):
        return snap

    val = snp.data["PartType0"]["access_snapshot2"]
    assert val is not None

    # test FieldContainer get()
    part0 = snp.data["PartType0"]
    assert part0.get("testfield", allow_derived=True).compute() == 0
    with pytest.raises(KeyError):
        part0.get("testfield", allow_derived=False)

    # test "all"
    @snp.register_field("all")
    def testfield_all(data, **kwargs):
        return True

    assert all([snp.data[k]["testfield_all"] for k in snp.data.keys()])
