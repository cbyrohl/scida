import numpy as np
import pytest

from scida.fields import FieldContainer
from scida.misc import (
    deepdictkeycopy,
    get_container_from_path,
    get_scalar,
    is_scalar,
    parse_size,
    str_is_float,
)


@pytest.mark.unit
class TestGetContainerFromPath:
    def test_retrieve_nested(self):
        root = FieldContainer()
        root.add_container("a")
        root["a"].add_container("b")

        res = get_container_from_path("a/b", root)
        assert res is root["a"]["b"]

    def test_slashes_handling(self):
        root = FieldContainer()
        root.add_container("a")
        root["a"].add_container("b")

        assert get_container_from_path("/a/b/", root) is root["a"]["b"]
        assert get_container_from_path("a//b", root) is root["a"]["b"]

    def test_create_missing(self):
        root = FieldContainer()
        res = get_container_from_path("a/b/c", root, create_missing=True)
        assert "a" in root._containers
        assert "b" in root["a"]._containers
        assert "c" in root["a"]["b"]._containers
        assert res is root["a"]["b"]["c"]

    def test_missing_raises(self):
        root = FieldContainer()
        with pytest.raises(ValueError) as excinfo:
            get_container_from_path("a/b", root, create_missing=False)
        assert "Container 'a' not found" in str(excinfo.value)


@pytest.mark.unit
class TestParseSize:
    """Test parse_size with inputs matching actual usage (dask chunk-size format).

    Note: parse_size has a known limitation where the number of numeric digits
    must equal the number of unit characters for correct parsing. Tests use
    inputs that satisfy this constraint, matching real-world usage like "128MiB".
    """

    def test_mebibytes(self):
        assert parse_size("128MiB") == 128 * 1024**2

    def test_gibibytes(self):
        assert parse_size("100GiB") == 100 * 1024**3

    def test_kilobytes(self):
        assert parse_size("10KB") == 10_000

    def test_megabytes(self):
        assert parse_size("10MB") == 10_000_000


@pytest.mark.unit
class TestIsScalar:
    def test_int(self):
        assert is_scalar(42) is True

    def test_float(self):
        assert is_scalar(3.14) is True

    def test_numpy_int(self):
        assert is_scalar(np.int64(5)) is True

    def test_numpy_float(self):
        assert is_scalar(np.float32(1.0)) is True

    def test_numpy_0d_array(self):
        assert is_scalar(np.array(5)) is True

    def test_numpy_1d_array(self):
        assert is_scalar(np.array([1, 2, 3])) is False

    def test_string(self):
        assert is_scalar("hello") is False

    def test_list(self):
        assert is_scalar([1, 2]) is False


@pytest.mark.unit
class TestGetScalar:
    def test_from_int(self):
        assert get_scalar(5) == 5.0

    def test_from_float(self):
        assert get_scalar(3.14) == 3.14

    def test_from_numpy_scalar(self):
        assert get_scalar(np.float64(2.5)) == 2.5

    def test_from_numpy_1d_array(self):
        arr = np.array([7, 8, 9])
        assert get_scalar(arr) == 7


@pytest.mark.unit
class TestStrIsFloat:
    def test_integer_string(self):
        assert str_is_float("42") is True

    def test_float_string(self):
        assert str_is_float("3.14") is True

    def test_scientific_notation(self):
        assert str_is_float("1e10") is True

    def test_non_numeric(self):
        assert str_is_float("hello") is False

    def test_empty_string(self):
        assert str_is_float("") is False


@pytest.mark.unit
class TestDeepdictkeycopy:
    def test_simple_nested(self):
        old = {"a": {"b": 1, "c": 2}, "d": 3}
        new = {}
        deepdictkeycopy(old, new)
        assert "a" in new
        assert isinstance(new["a"], dict)
        assert "b" not in new["a"]  # values not copied
        assert "d" not in new  # non-dict values not copied

    def test_deeply_nested(self):
        old = {"a": {"b": {"c": 1}}}
        new = {}
        deepdictkeycopy(old, new)
        assert "a" in new
        assert "b" in new["a"]
        assert isinstance(new["a"]["b"], dict)

    def test_empty_dict(self):
        old = {}
        new = {}
        deepdictkeycopy(old, new)
        assert new == {}
