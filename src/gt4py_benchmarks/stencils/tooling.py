"""
Tooling around gtscript stencils.

Enables:

    * Stencil wrapper classes that define the stencil but also
        subroutines and algorithmic parameters.
    * Logical grouping of subroutines into classes.
    * Distributing subroutines and grouping classes across modules.
    * Building many storages with the same backend, origin etc without repeating code.
"""
import abc

from gt4py import gtscript, storage


class StorageBuilder:
    """Convenience builder to store common config and avoid repetitive storage creation code."""

    def __init__(self, backend=None, default_origin=None, dtype=None):
        """Initialize with optional default values."""
        self._kwargs = {"backend": backend, "default_origin": default_origin, "dtype": dtype}

    def copy(self):
        """Create another Instance with the same configuration."""
        return self.__class__(**self._kwargs)

    def backend(self, backend):
        """Set the default backend."""
        self._kwargs["backend"] = backend
        return self

    def default_origin(self, origin):
        """Set the default origin."""
        self._kwargs["default_origin"] = origin
        return self

    def dtype(self, dtype):
        """Set the default dtype."""
        self._kwargs["dtype"] = dtype
        return self

    def shape(self, shape):
        """Set the default shape."""
        self._kwargs["shape"] = shape
        return self

    def from_array(self, *args, **kwargs):
        """Create a storage from a numpy array."""
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.from_array(*args, **keywords)

    def empty(self, *args, **kwargs):
        """Create a storage with uninitialized fields."""
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.empty(*args, **keywords)

    def zeros(self, *args, **kwargs):
        """Create a storage with all fields initialized to 0."""
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.zeros(*args, **keywords)

    def ones(self, *args, **kwargs):
        """Create a storage with all fields initialized to 1."""
        keywords = self._kwargs.copy()
        keywords.update(kwargs)
        return storage.ones(*args, **keywords)


class AbstractStencil(abc.ABC):
    """
    Stencil interface to aid grouping per-layer subroutines and composition.

    A stencil can expose subroutines to other stencils or use subroutines from other stencils
    or substencils. Management of the `externals` dictionary to build the stencil is automatic.

    Another use-case is encapsulating algorithm detail state or parameters.
    """

    def __init__(self, *, backend="debug", **kwargs):
        """Construct the stencil from a backend name and additional data."""
        self._backend = backend
        self._stencil = None

    @classmethod
    @abc.abstractmethod
    def stencil_definition(cls):
        """
        Return the stencil definition.

        define the stencil as a staticmethod in the class and return it from here.

        Subroutines defined as methods of this class are automatically injected into the externals.
        Additional externals can be injected by overriding the `build_externals` class method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def subroutines(cls):
        """List stencil subroutines implemented in this class."""
        pass

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Name the stencil for naming of subroutines externals when used by other stencils."""
        pass

    @classmethod
    def uses(cls):
        """List substencil classes used."""
        return []

    @classmethod
    def externals(cls) -> dict:
        """Build the externals dictionary."""
        ext_dict = {}
        for sub in cls.uses():
            ext_dict.update(sub.externals())
        ext_dict.update({sub.__name__: gtscript.function(sub) for sub in cls.subroutines()})
        return ext_dict

    def build(self, **kwargs):
        """Build the stencil, get ready to execute."""
        externals = self.externals()
        externals.update(kwargs.pop("externals", {}))
        self._stencil = gtscript.stencil(
            definition=self.stencil_definition(),
            backend=self._backend,
            externals=externals,
            **kwargs,
        )
        return self

    @property
    def backend(self):
        """Backend property."""
        return self._backend

    @abc.abstractmethod
    def copy_data(self, other):
        """Copy additional state (subclass hook)."""
        pass

    def copy_with_backend(self, backend):
        """Create a copy of the stencil (including internal state) but with another backend."""
        new = self.__class__(backend=backend)
        new.copy_data(self)
        return new

    def _call(self, *args, **kwargs):
        """Execute the internal gtscripts stencil."""
        self._stencil(*args, **kwargs)

    @property
    def stencil_obj(self):
        """Return compiled stencil object, only compiling the first time."""
        if self._stencil is None:
            self.build()
        return self._stencil

    def min_origin(self):
        """Get minimum origin for this stencil (assume `inp` input field is indicative of all)."""
        return tuple(max(i[0], i[1]) for i in self.stencil_obj.field_info["inp"].boundary)

    def storage_builder(self):
        """Create a preconfigured storage builder."""
        builder = StorageBuilder().backend(self.backend)
        if hasattr(self, "SCALAR_T"):
            builder = builder.dtype(self.SCALAR_T)
        return builder


class AbstractSubstencil:
    """
    A substencil defines one or more subroutines which logically belong together.

    Another substencil or stencil can declare usage of a substencil in it's `uses` method.
    The original substencil's subroutines will then be injected automatically into the `externals`
    of every stencil at the end of the `uses` chain.

    A substencil should never be used for holding any state.
    """

    @classmethod
    @abc.abstractmethod
    def name(cls):
        """Declare name."""
        return ""

    @classmethod
    def uses(cls):
        """List substencil classes used."""
        return []

    @classmethod
    def externals(cls):
        """Return an `externals` dictionary that can be used to compile stencils."""
        ext_dict = {}
        for sub in cls.uses():
            ext_dict.update(sub.externals())
        ext_dict.update(cls.subs())
        return ext_dict

    @classmethod
    def subs(cls):
        """Return a dictionary with subroutines, keyed on their name."""
        subroutines = (name for name in cls.__dict__ if name.startswith(cls.name()))
        cls_subs = {name: gtscript.function(getattr(cls, name)) for name in subroutines}
        return cls_subs


def using(globals, *substencils):
    """Pull required substencils' subroutines into global scope for gtscript."""

    def decorator(new_substencil):
        for sub in substencils:
            globals.update(sub.externals())
        return new_substencil

    return decorator
