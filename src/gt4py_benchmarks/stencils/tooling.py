import abc

from gt4py import gtscript, storage


class StorageBuilder:
    def __init__(self, backend=None, default_origin=None, dtype=None):
        self._kwargs = {"backend": backend, "default_origin": default_origin, "dtype": dtype}

    def copy(self):
        return self.__class__(**self._kwargs)

    def backend(self, backend):
        self._kwargs["backend"] = backend
        return self

    def default_origin(self, origin):
        self._kwargs["default_origin"] = origin
        return self

    def dtype(self, dtype):
        self._kwargs["dtype"] = dtype
        return self

    def shape(self, shape):
        self._kwargs["shape"] = shape
        return self

    def from_array(self, *args, **kwargs):
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.from_array(*args, **keywords)

    def empty(self, *args, **kwargs):
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.empty(*args, **keywords)

    def zeros(self, *args, **kwargs):
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.zeros(*args, **keywords)

    def ones(self, *args, **kwargs):
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.ones(*args, **keywords)


class AbstractStencil(abc.ABC):
    """
    Stencil interface to aid grouping per-layer subroutines and composition.

    Another use-case is encapsulating algorithm detail state.
    """

    def __init__(self, *, backend="debug", **kwargs):
        self._backend = backend
        self._stencil = None

    @staticmethod
    @abc.abstractmethod
    def stencil_definition(*args, **kwargs):
        """
        This is where the stencil is defined.

        Subroutines defined as methods of this class are automatically injected into the externals.
        Additional externals can be injected by overriding the `build_externals` class method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def subroutines(cls):
        """
        List stencil subroutines implemented in this class.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """
        Name the stencil for naming of subroutines externals when used by other stencils.
        """
        pass

    @classmethod
    def uses(cls):
        """List substencil classes used."""
        return []

    @classmethod
    def build_externals(cls) -> dict:
        """
        Build the externals dictionary
        """
        ext_dict = {}
        for sub in cls.uses():
            ext_dict.update(sub.externals())
        ext_dict.update({sub.__name__: gtscript.function(sub) for sub in cls.subroutines()})
        return ext_dict

    def build(self, **kwargs):
        """
        Build the stencil, get ready to execute.
        """
        externals = self.build_externals()
        externals.update(kwargs.pop("externals", {}))
        self._stencil = gtscript.stencil(
            definition=self.stencil_definition,
            backend=self._backend,
            externals=externals,
            **kwargs,
        )
        return self

    @property
    def backend(self):
        return self._backend

    @abc.abstractmethod
    def copy_data(self, other):
        """Subclass hook for copying additional data"""
        pass

    def copy_with_backend(self, backend):
        new = self.__class__(backend=backend)
        new.copy_data(self)
        return new

    def __call__(self, *args, **kwargs):
        """
        Execute the internal gtscripts stencil.
        """
        if self._stencil is None:
            self.build()
        self._stencil(*args, **kwargs)

    def storage_builder(self):
        builder = StorageBuilder().backend(self.backend)
        if hasattr(self, "SCALAR_T"):
            builder = builder.dtype(self.SCALAR_T)
        return builder


class AbstractSubstencil:
    @classmethod
    @abc.abstractmethod
    def name(cls):
        return ""

    @classmethod
    def uses(cls):
        """List substencil classes used."""
        return []

    @classmethod
    def externals(cls):
        ext_dict = {}
        for sub in cls.uses():
            ext_dict.update(sub.externals())
        ext_dict.update(cls.subs())
        return ext_dict

    @classmethod
    def subs(cls):
        subroutines = (name for name in cls.__dict__ if name.startswith(cls.name()))
        cls_subs = {name: gtscript.function(getattr(cls, name)) for name in subroutines}
        return cls_subs


def using(globals, *substencils):
    def decorator(new_substencil):
        for sub in substencils:
            globals.update(sub.externals())
        return new_substencil

    return decorator
