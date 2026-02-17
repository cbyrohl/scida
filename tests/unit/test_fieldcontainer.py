import dask.array as da
import numpy as np
import pytest

from scida.fields import FieldContainer, FieldRecipe, FieldType, walk_container


@pytest.mark.unit
class TestFieldContainerBasics:
    def test_empty_container(self):
        fc = FieldContainer()
        assert len(fc) == 0
        assert fc.fieldcount == 0
        assert fc.fieldlength is None
        assert list(fc.keys()) == []

    def test_set_and_get_field(self):
        fc = FieldContainer()
        arr = da.ones(10)
        fc["field1"] = arr
        assert "field1" in fc.keys()
        assert da.allclose(fc["field1"], arr)

    def test_fieldcount_excludes_internals(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc["uid"] = da.arange(10)  # uid is internal by default
        assert fc.fieldcount == 1

    def test_fieldcount_includes_recipes(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc.register_field(name="derived")(lambda container: da.zeros(10))
        assert fc.fieldcount == 2

    def test_fieldlength(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc["field2"] = da.ones(10)
        assert fc.fieldlength == 10

    def test_delitem_field(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        assert "field1" in fc.keys()
        del fc["field1"]
        assert "field1" not in fc.keys()

    def test_delitem_unknown_raises(self):
        fc = FieldContainer()
        with pytest.raises(KeyError):
            del fc["nonexistent"]

    def test_getitem_unknown_raises(self):
        fc = FieldContainer()
        with pytest.raises(KeyError):
            fc["nonexistent"]

    def test_repr(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        rep = repr(fc)
        assert "FieldContainer" in rep
        assert "fields=1" in rep

    def test_len(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(5)
        fc["field2"] = da.ones(5)
        assert len(fc) == 2


@pytest.mark.unit
class TestFieldContainerKeys:
    def test_keys_withgroups(self):
        fc = FieldContainer()
        fc.add_container("group1")
        fc["field1"] = da.ones(5)
        keys = fc.keys(withgroups=True)
        assert "group1" in keys
        assert "field1" in keys

    def test_keys_without_groups(self):
        fc = FieldContainer()
        fc.add_container("group1")
        fc["field1"] = da.ones(5)
        keys = fc.keys(withgroups=False)
        assert "group1" not in keys
        assert "field1" in keys

    def test_keys_without_recipes(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(5)
        fc.register_field(name="derived")(lambda container: da.zeros(5))
        keys_with = fc.keys(withrecipes=True)
        keys_without = fc.keys(withrecipes=False)
        assert "derived" in keys_with
        assert "derived" not in keys_without

    def test_keys_with_internal(self):
        fc = FieldContainer()
        fc["uid"] = da.arange(5)
        fc["field1"] = da.ones(5)
        keys_with = fc.keys(withinternal=True)
        keys_without = fc.keys(withinternal=False)
        assert "uid" in keys_with
        assert "uid" not in keys_without

    def test_keys_sorted(self):
        fc = FieldContainer()
        fc["zebra"] = da.ones(5)
        fc["alpha"] = da.ones(5)
        keys = fc.keys()
        assert keys == sorted(keys)


@pytest.mark.unit
class TestFieldContainerCopy:
    def test_copy_fields(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc2 = fc.copy()
        assert "field1" in fc2.keys()
        assert da.allclose(fc2["field1"], da.ones(10))

    def test_copy_independent(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc2 = fc.copy()
        fc2["field2"] = da.zeros(10)
        assert "field2" not in fc.keys()

    def test_copy_preserves_aliases(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc.add_alias("alias1", "field1")
        fc2 = fc.copy()
        assert "alias1" in fc2.aliases

    def test_copy_skeleton(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc.add_container("group1")
        fc["group1"]["inner_field"] = da.ones(5)
        skeleton = fc.copy_skeleton()
        assert "group1" in skeleton._containers
        assert "field1" not in skeleton._fields
        assert "inner_field" not in skeleton["group1"]._fields


@pytest.mark.unit
class TestFieldContainerContainers:
    def test_add_container_by_name(self):
        fc = FieldContainer()
        fc.add_container("group1")
        assert "group1" in fc._containers
        assert isinstance(fc["group1"], FieldContainer)

    def test_add_container_by_instance(self):
        fc = FieldContainer()
        child = FieldContainer(name="child")
        child["field1"] = da.ones(5)
        fc.add_container(child, deep=True)
        assert "child" in fc._containers
        assert "field1" in fc["child"].keys()

    def test_remove_container(self):
        fc = FieldContainer()
        fc.add_container("group1")
        fc.remove_container("group1")
        assert "group1" not in fc._containers

    def test_remove_container_unknown_raises(self):
        fc = FieldContainer()
        with pytest.raises(KeyError):
            fc.remove_container("nonexistent")

    def test_set_container(self):
        fc = FieldContainer()
        child = FieldContainer(name="child")
        fc["child"] = child
        assert "child" in fc._containers
        assert isinstance(fc["child"], FieldContainer)

    def test_nested_access(self):
        fc = FieldContainer()
        fc.add_container("group1")
        fc["group1"]["field1"] = da.ones(5)
        assert da.allclose(fc["group1"]["field1"], da.ones(5))

    def test_constructor_containers_list(self):
        fc = FieldContainer(containers=["group1", "group2"])
        assert "group1" in fc._containers
        assert "group2" in fc._containers


@pytest.mark.unit
class TestFieldContainerMerge:
    def test_merge_root_fields(self):
        fc1 = FieldContainer()
        fc1["field1"] = da.ones(5)
        fc2 = FieldContainer()
        fc2["field2"] = da.zeros(5)
        fc1.merge(fc2)
        assert "field1" in fc1._fields
        assert "field2" in fc1._fields

    def test_merge_overwrite(self):
        fc1 = FieldContainer()
        fc1["field1"] = da.ones(5)
        fc2 = FieldContainer()
        fc2["field1"] = da.zeros(5)
        fc1.merge(fc2, overwrite=True)
        assert da.allclose(fc1["field1"], da.zeros(5))

    def test_merge_nested_containers(self):
        fc1 = FieldContainer()
        fc1.add_container("group1")
        fc1["group1"]["a"] = da.ones(5)

        fc2 = FieldContainer()
        fc2.add_container("group1")
        fc2["group1"]["b"] = da.zeros(5)

        fc1.merge(fc2)
        assert "a" in fc1["group1"]._fields
        assert "b" in fc1["group1"]._fields

    def test_merge_type_error(self):
        fc = FieldContainer()
        with pytest.raises(TypeError):
            fc.merge("not a container")


@pytest.mark.unit
class TestFieldContainerAliases:
    def test_alias_get(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc.add_alias("alias1", "field1")
        assert da.allclose(fc["alias1"], da.ones(10))

    def test_alias_set(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(10)
        fc.add_alias("alias1", "field1")
        fc["alias1"] = da.zeros(10)
        assert da.allclose(fc["field1"], da.zeros(10))


@pytest.mark.unit
class TestFieldContainerInfo:
    def test_info_empty(self):
        fc = FieldContainer(name="root")
        info = fc.info()
        assert isinstance(info, str)

    def test_info_with_fields(self):
        fc = FieldContainer(name="root")
        fc["field1"] = da.ones(10)
        info = fc.info()
        assert "root" in info
        assert "fields: 1" in info

    def test_info_with_containers(self):
        fc = FieldContainer(name="root")
        fc.add_container("group1")
        fc["group1"]["field1"] = da.ones(5)
        info = fc.info()
        assert "group1" in info


@pytest.mark.unit
class TestFieldContainerDataFrame:
    def test_get_dataframe_1d(self):
        fc = FieldContainer()
        fc["x"] = da.arange(10)
        fc["y"] = da.ones(10)
        df = fc.get_dataframe()
        assert "x" in df.columns
        assert "y" in df.columns
        assert len(df) == 10

    def test_get_dataframe_2d(self):
        fc = FieldContainer()
        fc["coords"] = da.ones((10, 3))
        df = fc.get_dataframe()
        assert "coords0" in df.columns
        assert "coords1" in df.columns
        assert "coords2" in df.columns

    def test_get_dataframe_selected_fields(self):
        fc = FieldContainer()
        fc["x"] = da.arange(10)
        fc["y"] = da.ones(10)
        df = fc.get_dataframe(fields=["x"])
        assert "x" in df.columns
        assert "y" not in df.columns

    def test_get_dataframe_unknown_field_raises(self):
        fc = FieldContainer()
        fc["x"] = da.arange(10)
        with pytest.raises(ValueError):
            fc.get_dataframe(fields=["nonexistent"])
