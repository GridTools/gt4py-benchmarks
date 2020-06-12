"""Implement advection stencils ported from GTBench."""
from typing import Sequence

from numpy import float64
from gt4py.gtscript import Field, computation, interval, PARALLEL, BACKWARD, FORWARD

from gt4py_benchmarks.stencils import tridiagonal
from gt4py_benchmarks.stencils.tooling import AbstractStencil


def weights():
    """Initialize weights for flux calculations."""
    return (1.0 / 30.0, -1.0 / 4.0, 1.0, -1.0 / 3.0, -1.0 / 2.0, 1.0 / 20.0)


class Horizontal(AbstractStencil):
    """Horizontal advection stencil."""

    def __init__(self, *, dspace: Sequence[float64], backend="debug", **kwargs):
        """Construct from spacial resolution and backend name."""
        self.dx = float64(dspace[0])
        self.dy = float64(dspace[1])
        self.velocities = [float64(5), float64(-2), float64(0)]
        self.velocities = kwargs.pop("velocities", self.velocities)
        super().__init__(backend=backend)

    @classmethod
    def name(cls):
        """Declare the stencil name."""
        return "horizontal_advection"

    def copy_data(self, other):
        """Copy internal state from another instance."""
        self.dx = other.dx
        self.dy = other.dy
        self.velocities = other.velocities

    @classmethod
    def subroutines(cls):
        """Declare internal and external subroutines used in the stencil."""
        return [weights, cls.flux_v, cls.flux_u]

    @staticmethod
    def flux_u(*, data_in, u, dx):
        """Calculate the flux in x-direction."""
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
        """Calculate the flux in y direction."""
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

    @classmethod
    def stencil_definition(cls):
        """Return the stencil definition."""
        return cls._stencil_definition

    @staticmethod
    def _stencil_definition(
        data_out: Field[float64],
        data_in: Field[float64],
        *,
        dx: float64,
        dy: float64,
        u: float64,
        v: float64,
        dt: float64,
    ):
        """Calculate a horizontal advection iteration."""
        # pytype: disable=import-error
        from __externals__ import weights, flux_u, flux_v  # noqa: weights must be imported here

        # pytype: enable=import-error

        with computation(PARALLEL), interval(...):
            flux_x = flux_u(data_in=data_in, u=u, dx=dx)
            flux_y = flux_v(data_in=data_in, v=v, dy=dy)
            data_out = data_in - dt * (  # noqa: data_out is modified to store the result
                flux_x + flux_y
            )

    def __call__(self, out: Field[float64], inp: Field[float64], *, dt: float64):
        """Apply the compiled stencil."""
        u, v, w = self.velocities
        self._call(out, inp, dx=self.dx, dy=self.dy, u=u, v=v, dt=dt)


class Vertical(AbstractStencil):
    """Vertical advection stencil."""

    def __init__(self, *, dspace: Sequence[float64], backend="debug", **kwargs):
        """Construct from spacial resolution and backend name."""
        self.dz = float64(dspace[2])
        self.velocities = [float64(0), float64(0), float64(3)]
        self.velocities = kwargs.pop("velocities", self.velocities)
        super().__init__(backend=backend)

    @classmethod
    def name(cls):
        """Declare the stencil name."""
        return "vertical_advection"

    def copy_data(self, other):
        """Copy internal state from another instance."""
        self.dz = other.dz
        self.velocities = other.velocities

    @classmethod
    def subroutines(cls):
        """Declare subroutines used in the stencil."""
        return []

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [
            tridiagonal.PeriodicBackward1,
            tridiagonal.PeriodicForward2,
            tridiagonal.PeriodicBackward2,
            tridiagonal.Backward,
        ]

    @classmethod
    def stencil_definition(cls):
        """Return the stencil definition."""
        return cls._stencil_definition

    @staticmethod
    def _stencil_definition(
        data_out: Field[float64], data_in: Field[float64], *, w: float64, dz: float64, dt: float64
    ):
        """Run vertical advection iteration."""
        # pytype: disable=import-error
        from __externals__ import (
            periodic_backward1_0_m1,
            periodic_backward1_m1_last,
            periodic_forward2_0_1,
            periodic_backward2_m1_last,
            backward_0_m1,
        )

        # pytype: enable=import-error

        # turn w into a field (even though only constant velocity is implemented)
        with computation(PARALLEL), interval(...):
            w_field = w

        # stage advection w 0
        with computation(BACKWARD):
            with interval(-1, None):
                data_last = data_in
            with interval(0, -1):
                data_last = data_last[0, 0, 1]

        with computation(FORWARD):
            with interval(0, 1):
                data_first = data_in
            with interval(1, None):
                data_first = data_first[0, 0, -1]

        # stage advection w forward 1
        with computation(PARALLEL), interval(...):
            a = -0.25 * w / dz
            c = 0.25 * w / dz
            b = 1.0 / dt - a - c
            alpha = -a
            beta = a
            gamma = -b
        with computation(FORWARD):
            with interval(0, 1):
                d = (
                    (1.0 / dt * data_in)
                    - (0.25 * w_field[0, 0, 1] * (data_in[0, 0, 1] - data_in) / dz)
                    - (0.25 * w_field * (data_in - data_last) / dz)
                )
                b = b - gamma
                c = c / b
                d = d / b
            with interval(1, -1):
                d = (
                    (1.0 / dt * data_in)
                    - (0.25 * w_field[0, 0, 1] * (data_in[0, 0, 1] - data_in) / dz)
                    - (0.25 * w_field * (data_in - data_in[0, 0, -1]) / dz)
                )
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
            with interval(-1, None):
                d = (
                    (1.0 / dt * data_in)
                    - (0.25 * w_field * (data_first - data_in) / dz)
                    - (0.25 * w_field * (data_in - data_in[0, 0, -1]) / dz)
                )
                b = b - alpha * beta / gamma
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)

        # stage advection w backward 1
        with computation(BACKWARD):
            with interval(-1, None):
                x = periodic_backward1_m1_last(d)
            with interval(0, -1):
                x = periodic_backward1_0_m1(x, c, d)

        # stage advection w forward 2
        with computation(FORWARD):
            with interval(0, 1):
                c, d = periodic_forward2_0_1(a, b, c, d, alpha, gamma)
            with interval(1, -1):
                d = 0
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
            with interval(-1, None):
                d = alpha
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)

        # stage advection w backward 2
        with computation(BACKWARD):
            with interval(-1, None):
                z, z_top, x_top = periodic_backward2_m1_last(d=d, x=x)
            with interval(1, -1):
                z_top = z_top[0, 0, 1]
                x_top = x_top[0, 0, 1]
                z = backward_0_m1(out=z, c=c, d=d)
            with interval(0, 1):
                z_top = z_top[0, 0, 1]
                x_top = x_top[0, 0, 1]
                z = backward_0_m1(out=z, c=c, d=d)
                fact = (x + beta * x_top / gamma) / (1.0 + z + beta * z_top / gamma)
        with computation(FORWARD), interval(1, None):
            fact = fact[0, 0, -1]

        # stage advection 3
        with computation(PARALLEL), interval(...):
            data_out = x - fact * z  # noqa: data_out is modified to store the result

    def __call__(self, out: Field[float64], inp: Field[float64], *, dt: float64):
        """Apply vertical advection stencil."""
        u, v, w = self.velocities
        self._call(out, inp, dz=self.dz, w=w, dt=dt)


class Full(AbstractStencil):
    """Full Advection stepper."""

    def __init__(self, *, dspace: Sequence[float64], backend="debug", **kwargs):
        """Construct from spacial resolution and backend name."""
        self.dspace = dspace
        self.velocities = [float64(0), float64(0), float64(3)]
        self.velocities = kwargs.pop("velocities", self.velocities)
        super().__init__(backend=backend)

    def copy_data(self, other):
        """Copy internal state from another instance."""
        self.dspace = other.dspace
        self.velocities = other.velocities

    @classmethod
    def name(cls):
        """Declare stencil name."""
        return "full_advection"

    @classmethod
    def subroutines(cls):
        """Declare subroutines used in the stencil."""
        return []

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [
            tridiagonal.PeriodicBackward1,
            tridiagonal.PeriodicForward2,
            tridiagonal.PeriodicBackward2,
            tridiagonal.Backward,
            Horizontal,
        ]

    def __call__(self, out: Field[float64], inp: Field[float64], *, dt: float64):
        """Apply vertical advection stencil."""
        u, v, w = self.velocities
        dx, dy, dz = self.dspace
        self._call(out, inp, dx=dx, dy=dy, dz=dz, u=u, v=v, w=w, dt=dt)

    @staticmethod
    def _stencil_definition(
        data_out: Field[float64],
        data_in: Field[float64],
        *,
        u: float64,
        v: float64,
        w: float64,
        dx: float64,
        dy: float64,
        dz: float64,
        dt: float64,
    ):
        """Run vertical advection iteration."""
        # pytype: disable=import-error
        from __externals__ import (
            periodic_backward1_0_m1,
            periodic_backward1_m1_last,
            periodic_forward2_0_1,
            periodic_backward2_m1_last,
            backward_0_m1,
            flux_u,
            flux_v,
        )

        # pytype: enable=import-error

        # turn w into a field (even though only constant velocity is implemented)
        with computation(PARALLEL), interval(...):
            w_field = w

        # stage advection w 0
        with computation(BACKWARD):
            with interval(-1, None):
                data_last = data_in
            with interval(0, -1):
                data_last = data_last[0, 0, 1]

        with computation(FORWARD):
            with interval(0, 1):
                data_first = data_in
            with interval(1, None):
                data_first = data_first[0, 0, -1]

        # stage advection w forward 1
        with computation(PARALLEL), interval(...):
            a = -0.25 * w / dz
            c = 0.25 * w / dz
            b = 1.0 / dt - a - c
            alpha = -a
            beta = a
            gamma = -b
        with computation(FORWARD):
            with interval(0, 1):
                d = (
                    (1.0 / dt * data_in)
                    - (0.25 * w_field[0, 0, 1] * (data_in[0, 0, 1] - data_in) / dz)
                    - (0.25 * w_field * (data_in - data_last) / dz)
                )
                b = b - gamma
                c = c / b
                d = d / b
            with interval(1, -1):
                d = (
                    (1.0 / dt * data_in)
                    - (0.25 * w_field[0, 0, 1] * (data_in[0, 0, 1] - data_in) / dz)
                    - (0.25 * w_field * (data_in - data_in[0, 0, -1]) / dz)
                )
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
            with interval(-1, None):
                d = (
                    (1.0 / dt * data_in)
                    - (0.25 * w_field * (data_first - data_in) / dz)
                    - (0.25 * w_field * (data_in - data_in[0, 0, -1]) / dz)
                )
                b = b - alpha * beta / gamma
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)

        # stage advection w backward 1
        with computation(BACKWARD):
            with interval(-1, None):
                x = periodic_backward1_m1_last(d)
            with interval(0, -1):
                x = periodic_backward1_0_m1(x, c, d)

        # stage advection w forward 2
        with computation(FORWARD):
            with interval(0, 1):
                c, d = periodic_forward2_0_1(a, b, c, d, alpha, gamma)
            with interval(1, -1):
                d = 0
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
            with interval(-1, None):
                d = alpha
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)

        # stage advection w backward 2
        with computation(BACKWARD):
            with interval(-1, None):
                z, z_top, x_top = periodic_backward2_m1_last(d=d, x=x)
            with interval(1, -1):
                z_top = z_top[0, 0, 1]
                x_top = x_top[0, 0, 1]
                z = backward_0_m1(out=z, c=c, d=d)
            with interval(0, 1):
                z_top = z_top[0, 0, 1]
                x_top = x_top[0, 0, 1]
                z = backward_0_m1(out=z, c=c, d=d)
                fact = (x + beta * x_top / gamma) / (1.0 + z + beta * z_top / gamma)
        with computation(FORWARD), interval(1, None):
            fact = fact[0, 0, -1]

        # stage runge-kutta advection 3
        with computation(PARALLEL), interval(...):
            v_out = x - fact * z
            flux_x = flux_u(data_in=data_in, u=u, dx=dx)
            flux_y = flux_v(data_in=data_in, v=v, dy=dy)
            data_out = (  # noqa: data_out is modified to store the result
                data_in - dt * (flux_x + flux_y) + (v_out - data_in)
            )
