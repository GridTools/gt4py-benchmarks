from numpy import float64, array
from functools import update_wrapper, partial
from typing import Sequence

from gt4py.gtscript import (
    Field,
    computation,
    interval,
    FORWARD,
    BACKWARD,
    PARALLEL,
    stencil,
    function,
)

from gt4py_benchmarks.config import GT_BACKEND, STENCIL_VERBOSE
from gt4py_benchmarks.stencils import tridiagonal
from gt4py_benchmarks.stencils.tooling import AbstractStencil, AbstractSubstencil, using


DTYPE = float64
_F64 = Field[DTYPE]
STENCIL = update_wrapper(partial(stencil, backend=GT_BACKEND, verbose=STENCIL_VERBOSE), stencil)


class Horizontal(AbstractStencil):
    """Horizontal diffusion stencil."""

    SCALAR_T = float64
    FIELD_T = Field[SCALAR_T]

    def __init__(self, *, dspace: Sequence[SCALAR_T], coeff: SCALAR_T, backend="debug", **kwargs):
        super().__init__(backend=backend)
        self.dx = self.SCALAR_T(dspace[0])
        self.dy = self.SCALAR_T(dspace[1])
        self.coeff = coeff

    def copy_data(self, other):
        self.dx = other.dx
        self.dy = other.dy
        self.coeff = other.coeff

    @classmethod
    def name(cls):
        return "horizontal"

    @classmethod
    def subroutines(cls):
        return []

    @staticmethod
    def stencil_definition(
        out: FIELD_T, inp: FIELD_T, *, dx: SCALAR_T, dy: SCALAR_T, dt: SCALAR_T, coeff: SCALAR_T
    ):
        with computation(PARALLEL), interval(...):
            w_0 = -1.0 / 90.0
            w_1 = 5.0 / 36.0
            w_2 = -49.0 / 36.0
            w_3 = 49.0 / 36.0
            w_4 = -5.0 / 36.0
            w_5 = 1.0 / 90.0
            flx_x0 = (
                w_0 * inp[-3, 0]
                + w_1 * inp[-2, 0]
                + w_2 * inp[-1, 0]
                + w_3 * inp[0, 0]
                + w_4 * inp[1, 0]
                + w_5 * inp[2, 0]
            ) / dx
            flx_x1 = (
                w_0 * inp[-2, 0]
                + w_1 * inp[-1, 0]
                + w_2 * inp[0, 0]
                + w_3 * inp[1, 0]
                + w_4 * inp[2, 0]
                + w_5 * inp[3, 0]
            ) / dx
            flx_y0 = (
                w_0 * inp[0, -3]
                + w_1 * inp[0, -2]
                + w_2 * inp[0, -1]
                + w_3 * inp[0, 0]
                + w_4 * inp[0, 1]
                + w_5 * inp[0, 2]
            ) / dy
            flx_y1 = (
                w_0 * inp[0, -2]
                + w_1 * inp[0, -1]
                + w_2 * inp[0, 0]
                + w_3 * inp[0, 1]
                + w_4 * inp[0, 2]
                + w_5 * inp[0, 3]
            ) / dy

            flx_x0_tmp = flx_x0 * (0.0 if inp - inp[-1, 0] < 0.0 else flx_x0)
            flx_x0 = flx_x0_tmp
            flx_x1_tmp = flx_x1 * (0.0 if inp[1, 0] - inp < 0.0 else flx_x1)
            flx_x1 = flx_x1_tmp
            flx_y0_tmp = flx_y0 * (0.0 if inp - inp[0, -1] < 0.0 else flx_y0)
            flx_y0 = flx_y0_tmp
            flx_y1_tmp = flx_y1 * (0.0 if inp[0, 1] - inp < 0.0 else flx_y1)
            flx_y1 = flx_y1_tmp

            out = inp + coeff * dt * ((flx_x1 - flx_x0) / dx + (flx_y1 - flx_y0) / dy)

    def __call__(self, out: FIELD_T, inp: FIELD_T, *, dt: SCALAR_T):
        super().__call__(out, inp, dx=self.dx, dy=self.dy, coeff=self.coeff, dt=dt)


class DiffusionW0(AbstractSubstencil):
    """
    Stage Diffusion W0.

    Usage in stencil::

        with computation(PARALLEL), interval(-1, None):
            data_top = diffusion_w0_m1_last(data)

    Used to copy the last layer of data into data_top.

    Args:
        * data: input field
        * data_top: (only in original) in/out field, storage_ij_t
    """

    @classmethod
    def name(cls):
        return "diffusion_w0"

    @staticmethod
    def diffusion_w0_m1_last(data):
        return data


@using(globals(), tridiagonal.PeriodicForward1)
class DiffusionWF1(AbstractSubstencil):
    """
    Diffusion W Forward 1 substencil.

    Args:
        * coeff, dz, dt: input scalars
        * data: input field
        * data_tmp: in/out field
        * a, b, c, d: pure output fields (ijk)
        * alpha, beta, gamma: pure output fields (ij)
    """

    @classmethod
    def name(cls):
        return "diffusion_wf1"

    @classmethod
    def uses(cls):
        return [tridiagonal.PeriodicForward1]

    @staticmethod
    def diffusion_wf1_0_1(data, data_tmp, *, dz, dt, coeff):
        c = -coeff / (2.0 * dz * dz)
        a = c
        b = 1.0 / dt - a - c
        d = 1.0 / dt * data + 0.5 * coeff * (data_tmp - 2.0 * data + data[0, 0, 1]) / (dz * dz)
        beta = -coeff / (2.0 * dz * dz)
        alpha = beta
        gamma = -b

        ## inlining periodic_forward1_0_1 & forward_0_1:
        ## b, c, d = periodic_forward1_0_1(a, b, c, d, alpha, beta, gamma)
        b = b - gamma
        c = c / b
        d = d / b
        ## end inlining

        data_tmp = data

        return alpha, beta, gamma, a, b, c, d, data_tmp

    @staticmethod
    def diffusion_wf1_1_m1(data, data_tmp, *, a, b, c, d, alpha, beta, gamma, dz, dt, coeff):
        c = -coeff / (2.0 * dz * dz)
        a = c
        b = 1.0 / dt - a - c
        d = 1.0 / dt * data + 0.5 * coeff * (data[0, 0, -1] - 2.0 * data + data[0, 0, 1]) / (
            dz * dz
        )

        ## inlining periodic_forward1_1_m1 & forward_1_last:
        ## b, c, d = periodic_forward1_1_m1(a, b, c, d, alpha, beta, gamma)
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
        ## end inlining

        return a, b, c, d

    @staticmethod
    def diffusion_wf1_m1_last(data, data_tmp, *, a, b, c, d, alpha, beta, gamma, dz, dt, coeff):
        c = -coeff / (2) * dz * dz
        a = c
        b = 1.0 / dt - a - c
        d = 1.0 / dt * data + 0.5 * coeff * (data[0, 0, -1] - 2.0 * data + data_tmp) / (dz * dz)

        ## inlining periodic_forward1_1_m1 & forward_1_last:
        ## b, c, d = periodic_forward1_m1_last(a, b, c, d, alpha, beta, gamma)
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
        ## end inlining

        return a, b, c, d


class Vertical(AbstractStencil):
    """Vertical diffusion stencil."""

    SCALAR_T = float64
    FIELD_T = Field[SCALAR_T]

    def __init__(self, *, dspace: Sequence[SCALAR_T], coeff: SCALAR_T, backend="debug", **kwargs):
        super().__init__(backend=backend)
        self.dz = dspace[2]
        self.coeff = coeff

    def copy_data(self, other):
        self.dz = other.dz
        self.coeff = other.coeff

    @classmethod
    def name(cls):
        return "vertical"

    @classmethod
    def subroutines(cls):
        return []

    @classmethod
    def uses(cls):
        return [
            DiffusionW0,
            DiffusionWF1,
            tridiagonal.PeriodicBackward1,
            tridiagonal.PeriodicForward2,
            tridiagonal.PeriodicBackward2,
            tridiagonal.Periodic3,
        ]

    @staticmethod
    def stencil_definition(data_out: _F64, data_in: _F64, *, dz: DTYPE, dt: DTYPE, coeff: DTYPE):
        from __externals__ import (
            diffusion_w0_m1_last,
            diffusion_wf1_0_1,
            diffusion_wf1_1_m1,
            diffusion_wf1_m1_last,
            periodic_backward1_0_m1,
            periodic_backward1_m1_last,
            periodic_forward2_0_1,
            periodic_forward2_1_m1,
            periodic_forward2_m1_last,
            periodic_backward2_0_1,
            periodic_backward2_1_m1,
            periodic_backward2_m1_last,
            periodic3_full,
        )

        with computation(FORWARD), interval(-1, None):
            data_in_tmp = diffusion_w0_m1_last(data_in)

        with computation(FORWARD):
            with interval(0, 1):
                alpha, beta, gamma, a, b, c, d, data_tmp = diffusion_wf1_0_1(
                    data_in, data_in_tmp, dz=dz, dt=dt, coeff=coeff
                )
            with interval(1, -1):
                a, b, c, d = diffusion_wf1_1_m1(
                    data_in,
                    data_tmp,
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    dz=dz,
                    dt=dt,
                    coeff=coeff,
                )
            with interval(1, None):
                a, b, c, d = diffusion_wf1_m1_last(
                    data_in,
                    data_tmp,
                    a=a,
                    b=b,
                    c=c,
                    d=d,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    dz=dz,
                    dt=dt,
                    coeff=coeff,
                )

        with computation(BACKWARD):
            with interval(-1, None):
                x = periodic_backward1_m1_last(d)
            with interval(0, -1):
                x = periodic_backward1_0_m1(x, c, d)

        with computation(FORWARD):
            with interval(0, 1):
                c, d = periodic_forward2_0_1(a, b, c, d, alpha, gamma)
            with interval(1, -1):
                c, d = periodic_forward2_1_m1(a, b, c, d, alpha, gamma)
            with interval(-1, None):
                c, d = periodic_forward2_m1_last(a, b, c, d, alpha, gamma)

        with computation(BACKWARD):
            with interval(-1, None):
                z, z_top, x_top = periodic_backward2_m1_last(d, x)
            with interval(1, -1):
                z = periodic_backward2_1_m1(z, c, d)
            with interval(0, 1):
                z, fact = periodic_backward2_0_1(z, c, d, x, beta, gamma, z_top, x_top)

        with computation(PARALLEL), interval(...):
            data_out = periodic3_full(x, z, fact)

    def __call__(self, out: FIELD_T, inp: FIELD_T, *, dt: SCALAR_T):
        super().__call__(out, inp, dz=self.dz, coeff=self.coeff, dt=dt)
