import numpy as np
import pytest

from scida.helpers_misc import (
    get_args,
    get_kwargs,
    hash_path,
    make_serializable,
    parse_humansize,
    sprint,
)


@pytest.mark.unit
class TestHashPath:
    def test_deterministic(self):
        assert hash_path("/some/path") == hash_path("/some/path")

    def test_different_paths(self):
        assert hash_path("/path/a") != hash_path("/path/b")

    def test_strips_trailing_slash(self):
        assert hash_path("/some/path/") == hash_path("/some/path")

    def test_length(self):
        result = hash_path("/some/path")
        assert len(result) == 16


@pytest.mark.unit
class TestMakeSerializable:
    def test_numpy_array(self):
        result = make_serializable(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_scalar(self):
        result = make_serializable(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_bytes(self):
        result = make_serializable(b"hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_plain_values(self):
        assert make_serializable(42) == 42
        assert make_serializable("hello") == "hello"
        assert make_serializable(3.14) == 3.14


@pytest.mark.unit
class TestGetKwargs:
    def test_simple_function(self):
        def func(a, b=1, c="hello"):
            pass

        kwargs = get_kwargs(func)
        assert kwargs == {"b": 1, "c": "hello"}

    def test_no_kwargs(self):
        def func(a, b):
            pass

        kwargs = get_kwargs(func)
        assert kwargs == {}

    def test_all_kwargs(self):
        def func(a=1, b=2):
            pass

        kwargs = get_kwargs(func)
        assert kwargs == {"a": 1, "b": 2}


@pytest.mark.unit
class TestGetArgs:
    def test_simple_function(self):
        def func(a, b, c=1):
            pass

        args = get_args(func)
        assert args == ["a", "b"]

    def test_no_args(self):
        def func(a=1, b=2):
            pass

        args = get_args(func)
        assert args == []


@pytest.mark.unit
class TestParseHumansize:
    def test_bytes(self):
        assert parse_humansize("100B") == 100

    def test_kibibytes(self):
        assert parse_humansize("1KiB") == 1024

    def test_mebibytes(self):
        assert parse_humansize("2MiB") == 2 * 1024**2

    def test_gibibytes(self):
        assert parse_humansize("1GiB") == 1024**3

    def test_with_space(self):
        assert parse_humansize("1 GiB") == 1024**3


@pytest.mark.unit
class TestSprint:
    def test_basic(self):
        result = sprint("hello", "world")
        assert result == "hello world\n"

    def test_custom_end(self):
        result = sprint("hello", end="")
        assert result == "hello"

    def test_multiple_args(self):
        result = sprint("a", "b", "c")
        assert "a b c" in result
