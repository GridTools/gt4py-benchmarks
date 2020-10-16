import abc
import typing

import numpy as np
import pydantic

from ...constants import HALO
from ...numerics.solver import SolverState
from ...numerics.stencil_backends.base import StencilBackend
from ...utils import registry


class SolveResult(typing.NamedTuple):
    error: float
    time: float


@registry.subclass_registry
class Runtime(pydantic.BaseModel, abc.ABC):
    stencil_backend: StencilBackend

    def init_state(self, discrete_solution, t=0):
        nx, ny, nz = discrete_solution.local_resolution
        i, j, k = np.mgrid[: nx + 2 * HALO, : ny + 2 * HALO, : nz + 1]
        dtype = self.stencil_backend.dtype

        data = discrete_solution.data(i, j, k, t).astype(dtype)
        u = discrete_solution.u(i, j, k, t).astype(dtype)
        v = discrete_solution.v(i, j, k, t).astype(dtype)
        w = discrete_solution.w(i, j, k, t).astype(dtype)

        return SolverState(
            discrete_solution.local_resolution,
            discrete_solution.delta,
            [self.stencil_backend.storage_from_array(data) for _ in range(3)],
            self.stencil_backend.storage_from_array(u),
            self.stencil_backend.storage_from_array(v),
            self.stencil_backend.storage_from_array(w),
        )

    def compute_error(self, state, discrete_solution, t):
        nx, ny, nz = discrete_solution.local_resolution
        i, j, k = np.ogrid[HALO : nx + HALO, HALO : ny + HALO, :nz]

        inner = (slice(HALO, nx + HALO), slice(HALO, ny + HALO), slice(0, nz))

        assert np.allclose(
            self.stencil_backend.array_from_storage(state.u)[inner],
            discrete_solution.u(i, j, k, t),
        )
        assert np.allclose(
            self.stencil_backend.array_from_storage(state.v)[inner],
            discrete_solution.v(i, j, k, t),
        )
        assert np.allclose(
            self.stencil_backend.array_from_storage(state.w)[inner],
            discrete_solution.w(i, j, k, t),
        )

        return np.amax(
            np.abs(
                self.stencil_backend.array_from_storage(state.data[0])[inner]
                - discrete_solution.data(i, j, k, t)
            )
        )

    @abc.abstractmethod
    def solve(self, analytical, stepper, global_resolution, tmax, dt) -> SolveResult:
        pass

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
