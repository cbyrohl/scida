import dask.array as da
import pytest

from astrodask.interface import FieldContainer
from tests.testdata_properties import require_testdata, require_testdata_path


def test_fieldcontainer_aliasing():
    fc = FieldContainer()
    assert fc.fields == {}
    fc.add_alias("test", "test2")
    with pytest.raises(KeyError):
        print(fc["test"])  # "test2" not defined yet
    fc["test"] = 1
    assert (
        "test2" in fc.fields
    )  # if we write to the alias, the original entry should be set


@require_testdata_path("interface", only=["TNG50-4_snapshot"])
def test_fieldtypes(testdatapath):
    from astrodask import load

    snp = load(testdatapath)
    gas = snp.data["PartType0"]
    print("Field count:", len(gas.fields))
    fnames = list(gas.keys(withrecipes=False))
    assert len(fnames) < 5, "not lazy loading fields (into recipes)"
    fnames = list(gas.keys())
    assert len(fnames) > 5, "not correctly considering recipes"
    assert "uid" not in gas.keys()


@require_testdata("interface", only=["TNG50-4_snapshot"])
def test_fields(testdata_interface):
    snp = testdata_interface
    print(snp.data)
    gas_field_register = snp.data["PartType0"].register_field()

    @gas_field_register
    def tfield(data, **kwargs):
        return da.zeros(1)

    @gas_field_register
    def tfield2(data, **kwargs):
        return data["tfield"] + 1

    @snp.register_field("PartType0", name="tfield3")
    def tfield_explicitname(data, **kwargs):
        return data["tfield"] + 2

    # first-order field
    val0 = snp.data["PartType0"]["tfield"].compute()
    assert val0 == 0
    # second-order field
    val1 = snp.data["PartType0"]["tfield2"].compute()
    assert val1 == 1
    # second-order field with custom name
    val2 = snp.data["PartType0"]["tfield3"].compute()
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
    assert part0.get("tfield", allow_derived=True).compute() == 0
    with pytest.raises(KeyError):
        part0.get("tfield", allow_derived=False)

    # test "all"
    @snp.register_field("all")
    def tfield_all(data, **kwargs):
        return True

    print(snp.data["PartType1"]["tfield_all"])
    assert all([snp.data[k]["tfield_all"] for k in snp.data.keys()])
