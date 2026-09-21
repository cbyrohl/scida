import pytest

from scida.interface import BaseDataset, create_datasetclass_with_mixins
from scida.interfaces.mixins.base import Mixin


class DummyMixinA(Mixin):
    """Dummy mixin for testing."""
    pass


class DummyMixinB(Mixin):
    """Another dummy mixin for testing."""
    pass


@pytest.mark.unit
def test_create_datasetclass_with_mixins_normal():
    """Test that normal mixin addition works correctly."""
    newcls = create_datasetclass_with_mixins(BaseDataset, [DummyMixinA])
    assert DummyMixinA in newcls.__bases__
    assert "BaseDatasetWithDummyMixinA" == newcls.__name__


@pytest.mark.unit
def test_create_datasetclass_with_mixins_multiple():
    """Test that adding multiple mixins works correctly."""
    newcls = create_datasetclass_with_mixins(BaseDataset, [DummyMixinA, DummyMixinB])
    assert DummyMixinA in newcls.__bases__
    assert DummyMixinB in newcls.__bases__
    assert "BaseDatasetWithDummyMixinAAndDummyMixinB" == newcls.__name__


@pytest.mark.unit
def test_create_datasetclass_with_mixins_duplicate_in_list():
    """Test that ValueError is raised when the same mixin appears multiple times in the list."""
    with pytest.raises(ValueError, match="appears multiple times"):
        create_datasetclass_with_mixins(BaseDataset, [DummyMixinA, DummyMixinA])


@pytest.mark.unit
def test_create_datasetclass_with_mixins_duplicate_in_hierarchy():
    """Test that ValueError is raised when a mixin already exists in the class hierarchy."""
    cls_with_mixin = create_datasetclass_with_mixins(BaseDataset, [DummyMixinA])
    with pytest.raises(ValueError, match="already present in the class hierarchy"):
        create_datasetclass_with_mixins(cls_with_mixin, [DummyMixinA])


@pytest.mark.unit
def test_create_datasetclass_with_mixins_different_mixin_to_mixed_class():
    """Test that adding a different mixin to an already-mixed class works."""
    cls_with_mixin_a = create_datasetclass_with_mixins(BaseDataset, [DummyMixinA])
    cls_with_both = create_datasetclass_with_mixins(cls_with_mixin_a, [DummyMixinB])
    assert DummyMixinB in cls_with_both.__bases__


@pytest.mark.unit
def test_create_datasetclass_with_mixins_empty_list():
    """Test that passing an empty list returns the original class."""
    result = create_datasetclass_with_mixins(BaseDataset, [])
    assert result is BaseDataset


@pytest.mark.unit
def test_create_datasetclass_with_mixins_none():
    """Test that passing None returns the original class."""
    result = create_datasetclass_with_mixins(BaseDataset, None)
    assert result is BaseDataset
