import functools
import typing

import numpy as np

from ..verification.analytical import AnalyticalSolution
from ..constants import HALO


class _DiscreteAnalyticalSolution(typing.NamedTuple):
    analytical: AnalyticalSolution
    global_resolution: typing.Tuple[int, int, int]
    local_resolution: typing.Tuple[int, int, int]
    local_offset: typing.Tuple[int, int, int]

    def _remap(self, f, staggered_z=False):
        staggered_offset = -0.5 if staggered_z else 0

        @functools.wraps(f)
        def remapped(i, j, k, t):
            return f(
                (i - HALO + self.local_offset[0]) * self.delta[0],
                (j - HALO + self.local_offset[1]) * self.delta[1],
                (k + self.local_offset[2] + staggered_offset) * self.delta[2],
                t,
            )

        return remapped

    @property
    def delta(self) -> typing.Tuple[float, float, float]:
        x, y, z = self.analytical.domain
        rx, ry, rz = self.global_resolution
        return x / rx, y / ry, z / rz

    @property
    def data(self) -> typing.Callable[[int, int, int, float], np.array]:
        return self._remap(self.analytical.data)

    @property
    def u(self) -> typing.Callable[[int, int, int, float], np.array]:
        return self._remap(self.analytical.u)

    @property
    def v(self) -> typing.Callable[[int, int, int, float], np.array]:
        return self._remap(self.analytical.v)

    @property
    def w(self) -> typing.Callable[[int, int, int, float], np.array]:
        return self._remap(self.analytical.w)


def discretize(analytical, global_resolution, local_resolution, local_offset):
    return _DiscreteAnalyticalSolution(
        analytical, global_resolution, local_resolution, local_offset
    )
