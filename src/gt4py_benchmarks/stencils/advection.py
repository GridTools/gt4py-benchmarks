from typing import Sequence

from numpy import float64
from gt4py.gtscript import Field

from gt4py_benchmarks.stencils.tooling import AbstractStencil


DTYPE = float64


def weights():
    return (1.0 / 30.0, -1.0 / 4.0, 1.0, -1.0 / 3.0, -1.0 / 2.0, 1.0 / 20.0)


class Horizontal(AbstractStencil):
    """Horizontal advection stencil."""

    SCALAR_T = DTYPE
    FIELD_T = Field[SCALAR_T]

    def __init__(self, *, dspace: Sequence[SCALAR_T], backend="debug", **kwargs):
        self.dx = self.SCALAR_T(dspace[0])
        self.dy = self.SCALAR_T(dspace[1])
        self.velocities = [self.SCALAR_T(5), self.SCALAR_T(-2), self.SCALAR_T(0)]
        self.velocities = kwargs.pop("velocities", self.velocities)
        super().__init__(backend=backend)

    @classmethod
    def name(cls):
        return "horizontal_advection"

    def copy_data(self, other):
        self.dx = other.dx
        self.dy = other.dy

    @classmethod
    def subroutines(cls):
        return [weights, cls.flux_v, cls.flux_u]

    @staticmethod
    def flux_u(*, data_in, u, dx):
        w0, w1, w2, w3, w4, w5 = weights()
        if_pos = (
            u
            * -(
                w0 * data_in[-3, 0, 0]
                + w1 * data_in[-2, 0, 0]
                + w2 * data_in[-1, 0, 0]
                + w3 * data_in
                + w4 * data_in[1, 0, 0]
                + w5 * data_in[2, 0, 0]
            )
            / dx
        )
        if_neg = (
            u
            * (
                w5 * data_in[-2, 0, 0]
                + w4 * data_in[-1, 0, 0]
                + w3 * data_in
                + w2 * data_in[1, 0, 0]
                + w1 * data_in[2, 0, 0]
                + w0 * data_in[3, 0, 0]
            )
            / dx
        )
        return if_pos if u > 0.0 else (if_neg if u < 0.0 else 0.0)

    @staticmethod
    def flux_v(*, data_in, v, dy):
        w0, w1, w2, w3, w4, w5 = weights()
        if_pos = (
            v
            * -(
                w0 * data_in[0, -3, 0]
                + w1 * data_in[0, -2, 0]
                + w2 * data_in[0, -1, 0]
                + w3 * data_in
                + w4 * data_in[0, 1, 0]
                + w5 * data_in[0, 2, 0]
            )
            / dy
        )
        if_neg = (
            v
            * (
                w5 * data_in[0, -2, 0]
                + w4 * data_in[0, -1, 0]
                + w3 * data_in
                + w2 * data_in[0, 1, 0]
                + w1 * data_in[0, 2, 0]
                + w0 * data_in[0, 3, 0]
            )
            / dy
        )
        return if_pos if v > 0.0 else (if_neg if v < 0.0 else 0.0)

    @staticmethod
    def stencil_definition(
        data_out: FIELD_T,
        data_in: FIELD_T,
        *,
        dx: SCALAR_T,
        dy: SCALAR_T,
        u: SCALAR_T,
        v: SCALAR_T,
        dt: SCALAR_T,
    ):
        from __externals__ import weights, flux_u, flux_v

        with computation(PARALLEL), interval(...):
            flux_x = flux_u(data_in=data_in, u=u, dx=dx)
            flux_y = flux_v(data_in=data_in, v=v, dy=dy)
            data_out = data_in - dt * (flux_x + flux_y)

    def __call__(self, out: FIELD_T, inp: FIELD_T, *, dt: SCALAR_T):
        u, v, w = self.velocities
        super().__call__(out, inp, dx=self.dx, dy=self.dy, u=u, v=v, dt=dt)
