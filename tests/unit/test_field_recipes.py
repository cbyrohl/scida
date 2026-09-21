import dask.array as da
import pytest

from scida.fields import DerivedFieldRecipe, FieldContainer, FieldRecipe, FieldType


@pytest.mark.unit
class TestFieldRecipe:
    def test_create_with_func(self):
        recipe = FieldRecipe("test", func=lambda c: da.ones(10))
        assert recipe.name == "test"
        assert recipe.type == FieldType.IO
        assert recipe.func is not None
        assert recipe.arr is None

    def test_create_with_arr(self):
        arr = da.ones(10)
        recipe = FieldRecipe("test", arr=arr)
        assert recipe.name == "test"
        assert recipe.arr is not None
        assert recipe.func is None

    def test_create_neither_raises(self):
        with pytest.raises(ValueError, match="Need to specify either func or arr"):
            FieldRecipe("test")

    def test_description_and_units(self):
        recipe = FieldRecipe(
            "test", func=lambda c: da.ones(10), description="A test field", units="m/s"
        )
        assert recipe.description == "A test field"
        assert recipe.units == "m/s"


@pytest.mark.unit
class TestDerivedFieldRecipe:
    def test_create(self):
        recipe = DerivedFieldRecipe("derived", func=lambda c: da.zeros(5))
        assert recipe.name == "derived"
        assert recipe.type == FieldType.DERIVED

    def test_is_subclass(self):
        recipe = DerivedFieldRecipe("derived", func=lambda c: da.zeros(5))
        assert isinstance(recipe, FieldRecipe)


@pytest.mark.unit
class TestRegisterField:
    def test_register_to_self(self):
        fc = FieldContainer()
        fc["x"] = da.arange(10)

        @fc.register_field(name="double_x")
        def double_x(container):
            return container["x"] * 2

        assert "double_x" in fc._fieldrecipes
        result = fc["double_x"]
        assert da.allclose(result, da.arange(10) * 2)

    def test_register_to_named_container(self):
        fc = FieldContainer()
        fc.add_container("group1")
        fc["group1"]["x"] = da.arange(5)

        @fc.register_field("group1", name="derived")
        def derived(container):
            return container["x"] * 3

        assert "derived" in fc["group1"]._fieldrecipes

    def test_register_to_all(self):
        fc = FieldContainer()
        fc.add_container("g1")
        fc.add_container("g2")

        @fc.register_field("all", name="common")
        def common(container):
            return da.ones(5)

        assert "common" in fc["g1"]._fieldrecipes
        assert "common" in fc["g2"]._fieldrecipes

    def test_register_uses_func_name(self):
        fc = FieldContainer()

        @fc.register_field()
        def my_field(container):
            return da.ones(5)

        assert "my_field" in fc._fieldrecipes
