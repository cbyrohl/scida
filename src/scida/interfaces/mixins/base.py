import abc

from scida.registries import mixin_type_registry


class Mixin:
    """Base mixin class for all mixins."""

    pass

    def __init_subclass__(cls, *args, **kwargs):
        """
        Register mixin subclass in registry.
        Parameters
        ----------
        args:
            (unused)
        kwargs:
            (unused)
        Returns
        -------
        None
        """
        super().__init_subclass__(*args, **kwargs)
        # only register if Mixin is base class
        allow_register = cls.__bases__ == (Mixin,)
        # only register immediate subclasses
        if allow_register:
            mixin_type_registry[cls.__name__] = cls

    @classmethod
    @abc.abstractmethod
    def validate(cls, metadata: dict, *args, **kwargs):
        """
        Validate whether the dataset is valid for this mixin.
        Parameters
        ----------
        metadata: dict
        args
        kwargs

        Returns
        -------
        bool

        """
        return False
