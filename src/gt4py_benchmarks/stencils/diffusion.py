"""
Diffusion stencils.

Ported from GTBench by Rico Häuselmann <ricoh@cscs.ch>.
"""
from typing import Sequence

from numpy import float64
from gt4py.gtscript import (
    Field,
    computation,
    interval,
    FORWARD,
    BACKWARD,
    PARALLEL,
)

from gt4py_benchmarks.stencils import tridiagonal
from gt4py_benchmarks.stencils.tooling import AbstractStencil, AbstractSubstencil, using


class Horizontal(AbstractStencil):
    """Horizontal diffusion stencil."""

    def __init__(self, *, dspace: Sequence[float64], coeff: float64, backend="debug", **kwargs):
        """Construct from backend name, spacial resolution and diffusion coefficient."""
        super().__init__(backend=backend)
        self.dx = float64(dspace[0])
        self.dy = float64(dspace[1])
        self.coeff = coeff

    def copy_data(self, other):
        """Copy internal state from another instance."""
        self.dx = other.dx
        self.dy = other.dy
        self.coeff = other.coeff

    @classmethod
    def name(cls):
        """Declare name."""
        return "horizontal_diffusion"

    @classmethod
    def subroutines(cls):
        """Declare subroutines."""
        return [cls.threshold]

    @staticmethod
    def threshold(val, *, diff):
        """Thresholding subroutine."""
        if val * diff < 0.0:
            return 0.0
        else:
            return val

    @classmethod
    def stencil_definition(cls):
        """Return the stencil definition."""
        return cls._stencil_definition

    @staticmethod
    def _stencil_definition(
        out: Field[float64],
        inp: Field[float64],
        *,
        dx: float64,
        dy: float64,
        dt: float64,
        coeff: float64,
    ):
        """Horizontal diffusion time step stencil."""
        from __externals__ import threshold  # noqa (required for subroutines to work)

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

            out = inp + (  # noqa (store result in out)
                coeff * dt * (((flx_x1 - flx_x0) / dx) + ((flx_y1 - flx_y0) / dy))
            )

    def __call__(self, out: Field[float64], inp: Field[float64], *, dt: float64):
        """Apply the horizontal diffusion timestep stencil."""
        self._call(out, inp, dx=self.dx, dy=self.dy, coeff=self.coeff, dt=dt)


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
        """Declare name."""
        return "diffusion_w0"

    @staticmethod
    def diffusion_w0_m1_last(data):
        """Copy data."""
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
        """Declare name."""
        return "copy_data_first"

    @staticmethod
    def copy_data_first_init(data):
        """Initiate the data slice copy, see :class:`CopyDataFirst` for usage."""
        return data

    @staticmethod
    def copy_data_first_forward(out):
        """
        Propagate data slice copy forward along the vertical direction.

        See :class:`CopyDataFirst` for usage.
        """
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
        """Declare name."""
        return "copy_data_last"

    @staticmethod
    def copy_data_last_init(data):
        """Initiate the data slice copy, see :class:`CopyDataLast` for usage."""
        return data

    @staticmethod
    def copy_data_last_backward(out):
        """
        Propagate data slice copy backwards along the vertical direction.

        See :class:`CopyDataLast` for usage.
        """
        return out[0, 0, 1]


class DWF1Helper(AbstractSubstencil):
    """
    Helper substencil for :class:`DiffusionWF1`.

    Required due to some interesting behaviour with nested subroutines.
    See :class:`DiffusionWF1` for more information.
    """

    @classmethod
    def name(cls):
        """Declare name."""
        return "dwf1_helper"

    @staticmethod
    def dwf1_helper_acbeta(*, dz, coeff):
        """Return the initial value for temporaries `a, c, beta`."""
        return -coeff / (2.0 * dz * dz)

    @staticmethod
    def dwf1_helper_c_forward(*, a, b, c):
        """Propagate initialization of temporary `c` along the vertical axis."""
        return c / (b - c[0, 0, -1] * a)

    @staticmethod
    def dwf1_helper_d_forward(*, a, b, c, d):
        """Propagate initialization of temporary `d` along the vertical axis."""
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
                b, c, d = diffusion_wf1_0_1(inp, data_last, a=a, b=b, c=c, alpha=alpha,
                                            beta=beta, gamma=gamma, dz=dz, dt=dt, coeff=coeff,)
            with interval(1, -1):
                d = diffusion_wf1_1_m1(inp, a=a, b=b, c=c, alpha=alpha,
                                       beta=beta, gamma=gamma, dz=dz, dt=dt, coeff=coeff,)
                c = dwf1_helper_c_forward(a=a, b=b, c=c)
                d = dwf1_helper_d_forward(a=a, b=b, c=c, d=d)
            with interval(-1, None):
                b, d = diffusion_wf1_m1_last(inp, data_first, a=a, b=b, c=c, alpha=alpha,
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
        """Declare name."""
        return "diffusion_wf1"

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [DWF1Helper, CopyDataFirst, tridiagonal.PeriodicForward1]

    @staticmethod
    def diffusion_wf1_init(*, dz, coeff, dt):
        """Initialize the temporaries `a, b, c, alpha, beta, gamma`."""
        a = dwf1_helper_acbeta(  # noqa (is imported from __externals__ in the stencil)
            dz=dz, coeff=coeff
        )
        c = a
        b = 1.0 / dt - a - c
        beta = a
        alpha = a
        gamma = -b
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def diffusion_wf1_0_1(data, data_last, *, a, b, c, alpha, beta, gamma, dz, dt, coeff):
        """Initialize the first layers of `b, c, d`."""
        d = 1.0 / dt * data + 0.5 * coeff * (data_last - 2.0 * data + data[0, 0, 1]) / (dz * dz)
        b, c, d = periodic_forward1_0_1(  # noqa (is imported from __externals__ in the stencil)
            a, b, c, d, alpha, beta, gamma
        )
        return b, c, d

    @staticmethod
    def diffusion_wf1_1_m1(data, *, a, b, c, alpha, beta, gamma, dz, dt, coeff):
        """Initialize the middle layers of `d`."""
        d = 1.0 / dt * data + 0.5 * coeff * (data[0, 0, -1] - 2.0 * data + data[0, 0, 1]) / (
            dz * dz
        )

        return d

    @staticmethod
    def diffusion_wf1_m1_last(data, data_first, *, a, b, c, alpha, beta, gamma, dz, dt, coeff):
        """Initialize the last vertical layer of `b` and `d`."""
        d = 1.0 / dt * data + 0.5 * coeff * (data[0, 0, -1] - 2.0 * data + data_first) / (dz * dz)
        b = b - (alpha * beta / gamma)

        return b, d


class Vertical(AbstractStencil):
    """Vertical diffusion stencil."""

    def __init__(self, *, dspace: Sequence[float64], coeff: float64, backend="debug", **kwargs):
        """Construct from backend name, spacial resolution and diffusion coefficient."""
        super().__init__(backend=backend)
        self.dz = dspace[2]
        self.coeff = coeff

    def copy_data(self, other):
        """Copy internal data from another instance."""
        self.dz = other.dz
        self.coeff = other.coeff

    @classmethod
    def name(cls):
        """Declare name."""
        return "vertical_diffusion"

    @classmethod
    def subroutines(cls):
        """Declare subroutines which are not in one of the substencils."""
        return []

    @classmethod
    def uses(cls):
        """Declare substencils."""
        return [
            CopyDataLast,
            CopyDataFirst,
            DiffusionWF1,
            tridiagonal.PeriodicBackward1,
            tridiagonal.PeriodicForward2,
            tridiagonal.PeriodicBackward2,
            tridiagonal.Periodic3,
        ]

    @classmethod
    def stencil_definition(cls):
        """Return the stencil definition."""
        return cls._stencil_definition

    @staticmethod
    def _stencil_definition(
        out: Field[float64], inp: Field[float64], *, dz: float64, dt: float64, coeff: float64,
    ):
        """Calculate vertical diffusion time step."""
        from __externals__ import (  # noqa (required for subroutines to work)
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

        # stage diffusion w 0
        with computation(BACKWARD):
            with interval(-1, None):
                data_last = copy_data_last_init(inp)
            with interval(0, -1):
                data_last = copy_data_last_backward(data_last)

        with computation(FORWARD):
            with interval(0, 1):
                data_first = inp
            with interval(1, None):
                data_first = data_first[0, 0, -1]

        # stage diffusion w forward 1
        with computation(PARALLEL), interval(...):
            a, b, c, alpha, beta, gamma = diffusion_wf1_init(dz=dz, coeff=coeff, dt=dt)
        with computation(FORWARD):
            with interval(0, 1):
                b, c, d = diffusion_wf1_0_1(
                    inp,
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
                    inp,
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
                    inp,
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

        # stage diffusion w backward 1
        with computation(BACKWARD):
            with interval(-1, None):
                x = periodic_backward1_m1_last(d)
            with interval(0, -1):
                x = periodic_backward1_0_m1(x, c, d)

        # stage diffusion w forward 2
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

        # stage diffusion w backward 2
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
            out = periodic3_full(x, z, fact)  # noqa (store result in out)

    def __call__(self, out: Field[float64], inp: Field[float64], *, dt: float64):
        """Calculate one vertical diffusion iteration."""
        self._call(out, inp, dz=self.dz, coeff=self.coeff, dt=dt)


class Full:
    """Full diffusion stepper."""

    def __init__(self, *, dspace: Sequence[float64], coeff: float64, backend="debug", **kwargs):
        """Construct from backend name, vertical resolution and diffusion coefficient."""
        self.backend = backend
        self.dspace = dspace
        self.coeff = coeff
        self.horizontal = Horizontal(dspace=dspace, coeff=coeff, backend=backend, **kwargs)
        self.vertical = Vertical(dspace=dspace, coeff=coeff, backend=backend, **kwargs)

    def __call__(self, out: Field[float64], inp: Field[float64], *, dt: float64):
        """Calculate one iteration of diffusion, storing the result in `out`."""
        self.horizontal(out, inp, dt=dt)
        self.vertical(out, inp, dt=dt)

    @classmethod
    def name(cls):
        """Declare the stencil name."""
        return "full_diffusion"

    def storage_builder(self):
        """Create a pre-configured storage builder."""
        return self.horizontal.storage_builder()

    def min_origin(self):
        """Get the minimum possible origin for input storages."""
        x, y, _ = self.horizontal.min_origin()
        _, _, z = self.vertical.min_origin()
        return (x, y, z)
