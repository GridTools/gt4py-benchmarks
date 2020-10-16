from . import base

from .gt4py_backend import GT4PyStencilBackend

REGISTRY = base.StencilBackend.subclass_registry()

__all__ = ["GT4PyStencilBackend", "REGISTRY"]
