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

    def from_array(self, *args, **kwargs):
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.from_array(*args, **keywords)


class AbstractStencil(abc.ABC):
    """
    Stencil interface to aid grouping per-layer subroutines and composition.

    Another use-case is encapsulating algorithm detail state.
    """

    def __init__(self, *, backend="debug"):
        self._backend = default_backend
        self._stencil = None
        self._externals = {}

    @abc.abstractstaticmethod
    def stencil_definition(*args, **kwargs):
        """
        This is where the stencil is defined.

        Subroutines defined as methods of this class are automatically injected into the externals.
        Additional externals can be injected by overriding the `build_externals` class method.
        """
        pass

    @abc.abstractclassmethod
    def subroutines(cls):
        """
        List stencil subroutines implemented in this class.
        """
        pass

    @abc.abstractclassmethod
    def name(cls) -> str:
        """
        Name the stencil for naming of subroutines externals when used by other stencils.
        """
        pass

    @classmethod
    def build_externals(cls) -> dict:
        """
        Build the externals dictionary
        """
        return {sub.__name__: gtscript.function(sub) for sub in cls.subroutines()}

    def build(self, **kwargs):
        """
        Build the stencil, get ready to execute.
        """
        self._stencil = gtscript.stencil(
            definition=self.stencil_definition, backend=backend, **kwargs
        )
        return self

    def backend(self, backend):
        self._backend = backend
        return self

    def __call__(self, *args, **kwargs):
        """
        Execute the internal gtscripts stencil.
        """
        if self._stencil is None:
            self.build()
        self._stencil(*args, **kwargs)

    def storage_builder(self):
        return StorageBuilder().backend(self.backend)
