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
from gt4py.storage import empty

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
        return [cls.threshold]

    @staticmethod
    def threshold(val, *, diff):
        if val * diff < 0.0:
            return 0.0
        else:
            return val

    @staticmethod
    def stencil_definition(
        out: FIELD_T, inp: FIELD_T, *, dx: SCALAR_T, dy: SCALAR_T, dt: SCALAR_T, coeff: SCALAR_T
    ):
        from __externals__ import threshold

        with computation(PARALLEL), interval(...):
            w_0 = -1.0 / 90.0
            w_1 = 5.0 / 36.0
            w_2 = -49.0 / 36.0
            w_3 = 49.0 / 36.0
            w_4 = -5.0 / 36.0
            w_5 = 1.0 / 90.0
            flx_x0 = (
                (w_0 * inp[-3, 0])
                + (w_1 * inp[-2, 0])
                + (w_2 * inp[-1, 0])
                + (w_3 * inp[0, 0])
                + (w_4 * inp[1, 0])
                + (w_5 * inp[2, 0])
            ) / dx
            flx_x1 = (
                (w_0 * inp[-2, 0])
                + (w_1 * inp[-1, 0])
                + (w_2 * inp[0, 0])
                + (w_3 * inp[1, 0])
                + (w_4 * inp[2, 0])
                + (w_5 * inp[3, 0])
            ) / dx
            flx_y0 = (
                (w_0 * inp[0, -3])
                + (w_1 * inp[0, -2])
                + (w_2 * inp[0, -1])
                + (w_3 * inp[0, 0])
                + (w_4 * inp[0, 1])
                + (w_5 * inp[0, 2])
            ) / dy
            flx_y1 = (
                (w_0 * inp[0, -2])
                + (w_1 * inp[0, -1])
                + (w_2 * inp[0, 0])
                + (w_3 * inp[0, 1])
                + (w_4 * inp[0, 2])
                + (w_5 * inp[0, 3])
            ) / dy

            flx_x0 = threshold(flx_x0, diff=inp - inp[-1, 0])
            flx_x1 = threshold(flx_x1, diff=(inp[1, 0] - inp))
            flx_y0 = threshold(flx_y0, diff=(inp - inp[0, -1]))
            flx_y1 = threshold(flx_y1, diff=(inp[0, 1] - inp))

            out = inp + (coeff * dt * (((flx_x1 - flx_x0) / dx) + ((flx_y1 - flx_y0) / dy)))

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


class CopyDataFirst(AbstractSubstencil):
    """
    Copy the 0th k-layer of `data` into each layer of `out`.

    This is used to simulate an 2D storage in `out`.

    Usage::

        with computation(FORWARD):
            with interval(0, 1):
                out = copy_data_first_init(data)
            with interval(1, None):
                out = copy_data_first_forward(out)

    Args::

        * input: data
        * in / out: out
    """

    @classmethod
    def name(cls):
        return "copy_data_first"

    @staticmethod
    def copy_data_first_init(data):
        return data

    @staticmethod
    def copy_data_first_forward(out):
        return out[0, 0, -1]


class CopyDataLast(AbstractSubstencil):
    """
    Copy the last k-layer of `data` into each layer of `out`.

    This is used to simulate an 2D storage in `out`.

    Usage::

        with computation(BACKWARD):
            with interval(-1, None):
                out = copy_data_last_init(data)
            with interval(0, -1):
                out = copy_data_first_backward(out)

    Args::

        * input: data
        * in / out: out
    """

    @classmethod
    def name(cls):
        return "copy_data_last"

    @staticmethod
    def copy_data_last_init(data):
        return data

    @staticmethod
    def copy_data_last_backward(out):
        return out[0, 0, 1]


class DWF1Helper(AbstractSubstencil):
    @classmethod
    def name(cls):
        return "dwf1_helper"

    @staticmethod
    def dwf1_helper_acbeta(*, dz, coeff):
        return -coeff / (2.0 * dz * dz)

    @staticmethod
    def dwf1_helper_c_forward(*, a, b, c):
        return c / (b - c[0, 0, -1] * a)

    @staticmethod
    def dwf1_helper_d_forward(*, a, b, c, d):
        return (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)


@using(globals(), DWF1Helper, CopyDataFirst, tridiagonal.PeriodicForward1)
class DiffusionWF1(AbstractSubstencil):
    """
    Diffusion W Forward 1 substencil.

    Usage::

        with computation(PARALLEL), interval(...):
            a, b, c, alpha, beta, gamma = diffusion_wf1_init(dz=dz, coeff=coeff, dt=dt)
        with computation(FORWARD):
            with interval(0, 1):
                b, c, d = diffusion_wf1_0_1(data_in, data_last, a=a, b=b, c=c, alpha=alpha,
                                            beta=beta, gamma=gamma, dz=dz, dt=dt, coeff=coeff,)
            with interval(1, -1):
                d = diffusion_wf1_1_m1(data_in, a=a, b=b, c=c, alpha=alpha,
                                       beta=beta, gamma=gamma, dz=dz, dt=dt, coeff=coeff,)
                c = dwf1_helper_c_forward(a=a, b=b, c=c)
                d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)
            with interval(-1, None):
                b, d = diffusion_wf1_m1_last(data_in, data_first, a=a, b=b, c=c, alpha=alpha,
                                             beta=beta, gamma=gamma, dz=dz, dt=dt, coeff=coeff,)
                c = dwf1_helper_c_forward(a=a, b=b, c=c)
                d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)

    Note that the usage has to be exactly like this. Calling the dwf1_helper_x_forward subroutines
    from diffusion_wf1_xxx subroutines has been proven to not be equivalent.

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
        return [DWF1Helper, CopyDataFirst, tridiagonal.PeriodicForward1]

    @staticmethod
    def diffusion_wf1_init(*, dz, coeff, dt):
        a = dwf1_helper_acbeta(dz=dz, coeff=coeff)
        c = a
        b = 1.0 / dt - a - c
        beta = a
        alpha = a
        gamma = -b
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def diffusion_wf1_0_1(data, data_last, *, a, b, c, alpha, beta, gamma, dz, dt, coeff):
        d = 1.0 / dt * data + 0.5 * coeff * (data_last - 2.0 * data + data[0, 0, 1]) / (dz * dz)
        b, c, d = periodic_forward1_0_1(a, b, c, d, alpha, beta, gamma)
        return b, c, d

    @staticmethod
    def diffusion_wf1_1_m1(data, *, a, b, c, alpha, beta, gamma, dz, dt, coeff):
        d = 1.0 / dt * data + 0.5 * coeff * (data[0, 0, -1] - 2.0 * data + data[0, 0, 1]) / (
            dz * dz
        )

        ## the following does not work, apparently because nested functions with data dependencies
        ## are buggy.
        ## inlining periodic_forward1_1_m1 & forward_1_last:
        # b, c, d = periodic_forward1_1_m1(a, b, c, d, alpha, beta, gamma)
        # c = dwf1_helper_c_forward(a=a, b=b, c=c)
        # d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)
        ## end inlining

        return d

    @staticmethod
    def diffusion_wf1_m1_last(data, data_first, *, a, b, c, alpha, beta, gamma, dz, dt, coeff):
        # data_tmp = copy_data_first_forward(data_tmp)

        d = 1.0 / dt * data + 0.5 * coeff * (data[0, 0, -1] - 2.0 * data + data_first) / (dz * dz)

        ## inlining periodic_forward1_1_m1 & forward_1_last:
        ## b, c, d = periodic_forward1_m1_last(a, b, c, d, alpha, beta, gamma)
        b = b - (alpha * beta / gamma)
        ## the following does not work, apparently because nested functions with data dependencies
        ## are buggy
        # c = dwf1_helper_c_forward(a=a, b=b, c=c)
        # d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)
        ## end inlining

        return b, d


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
            CopyDataLast,
            CopyDataFirst,
            DiffusionWF1,
            tridiagonal.PeriodicBackward1,
            tridiagonal.PeriodicForward2,
            tridiagonal.PeriodicBackward2,
            tridiagonal.Periodic3,
        ]

    @staticmethod
    def stencil_definition(
        data_out: FIELD_T, data_in: FIELD_T, *, dz: SCALAR_T, dt: SCALAR_T, coeff: SCALAR_T
    ):
        from __externals__ import (
            copy_data_last_init,
            copy_data_last_backward,
            copy_data_first_forward,
            diffusion_wf1_init,
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
            dwf1_helper_c_forward,
            dwf1_helper_d_forward,
            backward_0_m1,
        )

        ## stage diffusion w 0
        with computation(BACKWARD):
            with interval(-1, None):
                data_last = copy_data_last_init(data_in)
            with interval(0, -1):
                data_last = copy_data_last_backward(data_last)

        with computation(FORWARD):
            with interval(0, 1):
                data_first = data_in
            with interval(1, None):
                data_first = data_first[0, 0, -1]

        ## stage diffusion w forward 1
        with computation(PARALLEL), interval(...):
            a, b, c, alpha, beta, gamma = diffusion_wf1_init(dz=dz, coeff=coeff, dt=dt)
        with computation(FORWARD):
            with interval(0, 1):
                b, c, d = diffusion_wf1_0_1(
                    data_in,
                    data_last,
                    a=a,
                    b=b,
                    c=c,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    dz=dz,
                    dt=dt,
                    coeff=coeff,
                )
            with interval(1, -1):
                d = diffusion_wf1_1_m1(
                    data_in,
                    a=a,
                    b=b,
                    c=c,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    dz=dz,
                    dt=dt,
                    coeff=coeff,
                )
                c = dwf1_helper_c_forward(a=a, b=b, c=c)
                d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)
            with interval(-1, None):
                b, d = diffusion_wf1_m1_last(
                    data_in,
                    data_first,
                    a=a,
                    b=b,
                    c=c,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    dz=dz,
                    dt=dt,
                    coeff=coeff,
                )
                c = dwf1_helper_c_forward(a=a, b=b, c=c)
                d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)

        ## stage diffusion w backward 1
        with computation(BACKWARD):
            with interval(-1, None):
                x = periodic_backward1_m1_last(d)
            with interval(0, -1):
                x = periodic_backward1_0_m1(x, c, d)

        ## stage diffusion w forward 2
        with computation(FORWARD):
            with interval(0, 1):
                c, d = periodic_forward2_0_1(a, b, c, d, alpha, gamma)
            with interval(1, -1):
                d = 0
                c = dwf1_helper_c_forward(a=a, b=b, c=c)
                d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)
            with interval(-1, None):
                d = alpha
                c = dwf1_helper_c_forward(a=a, b=b, c=c)
                d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)

        ## stage diffusion w backward 2
        with computation(BACKWARD):
            with interval(-1, None):
                z, z_top, x_top = periodic_backward2_m1_last(d=d, x=x)
            with interval(1, -1):
                z_top = copy_data_last_backward(z_top)
                x_top = copy_data_last_backward(x_top)
                z = backward_0_m1(out=z, c=c, d=d)
            with interval(0, 1):
                z_top = copy_data_last_backward(z_top)
                x_top = copy_data_last_backward(x_top)
                z = backward_0_m1(out=z, c=c, d=d)
                fact = (x + beta * x_top / gamma) / (1.0 + z + beta * z_top / gamma)
        with computation(FORWARD), interval(1, None):
            fact = copy_data_first_forward(fact)

        with computation(PARALLEL), interval(...):
            data_out = periodic3_full(x, z, fact)

    def __call__(self, out: FIELD_T, inp: FIELD_T, *, dt: SCALAR_T):
        super().__call__(out, inp, dz=self.dz, coeff=self.coeff, dt=dt)
