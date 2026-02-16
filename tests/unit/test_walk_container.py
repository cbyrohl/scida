import dask.array as da
import pytest

from scida.fields import FieldContainer, walk_container


@pytest.mark.unit
class TestWalkContainer:
    def test_walk_fields(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(5)
        fc["field2"] = da.zeros(5)
        visited = []
        walk_container(fc, handler_field=lambda entry, path, **kw: visited.append(path))
        assert len(visited) == 2
        assert "/field1" in visited
        assert "/field2" in visited

    def test_walk_nested(self):
        fc = FieldContainer()
        fc.add_container("group1")
        fc["group1"]["inner"] = da.ones(3)
        fc["top"] = da.zeros(3)
        field_paths = []
        group_paths = []
        walk_container(
            fc,
            handler_field=lambda entry, path, **kw: field_paths.append(path),
            handler_group=lambda entry, path: group_paths.append(path),
        )
        assert "/top" in field_paths
        assert "/group1/inner" in field_paths
        assert "/group1" in group_paths

    def test_walk_no_handlers(self):
        fc = FieldContainer()
        fc["field1"] = da.ones(5)
        # should not raise even with no handlers
        walk_container(fc)
