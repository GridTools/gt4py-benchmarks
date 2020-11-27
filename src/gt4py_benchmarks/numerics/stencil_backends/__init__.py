from . import base

from .gtbench_backend import GTBenchStencilBackend
from .gt4py_backend import GT4PyStencilBackend

REGISTRY = base.StencilBackend.subclass_registry()

__all__ = ["GTBenchStencilBackend", "GT4PyStencilBackend", "REGISTRY"]
