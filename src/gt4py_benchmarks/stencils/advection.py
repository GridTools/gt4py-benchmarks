"""Implement advection stencils ported from GTBench."""
from typing import Sequence

from numpy import float64
from gt4py.gtscript import Field, computation, interval, PARALLEL, BACKWARD, FORWARD

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
    def flux_u(*, inp, u, dx):
        """Calculate the flux in x-direction."""
        w0, w1, w2, w3, w4, w5 = weights()
        if_pos = (
            u
            * -(
                w0 * inp[-3, 0, 0]
                + w1 * inp[-2, 0, 0]
                + w2 * inp[-1, 0, 0]
                + w3 * inp
                + w4 * inp[1, 0, 0]
                + w5 * inp[2, 0, 0]
            )
            / dx
        )
        if_neg = (
            u
            * (
                w5 * inp[-2, 0, 0]
                + w4 * inp[-1, 0, 0]
                + w3 * inp
                + w2 * inp[1, 0, 0]
                + w1 * inp[2, 0, 0]
                + w0 * inp[3, 0, 0]
            )
            / dx
        )
        return if_pos if u > 0.0 else (if_neg if u < 0.0 else 0.0)

    @staticmethod
    def flux_v(*, inp, v, dy):
        """Calculate the flux in y direction."""
        w0, w1, w2, w3, w4, w5 = weights()
        if_pos = (
            v
            * -(
                w0 * inp[0, -3, 0]
                + w1 * inp[0, -2, 0]
                + w2 * inp[0, -1, 0]
                + w3 * inp
                + w4 * inp[0, 1, 0]
                + w5 * inp[0, 2, 0]
            )
            / dy
        )
        if_neg = (
            v
            * (
                w5 * inp[0, -2, 0]
                + w4 * inp[0, -1, 0]
                + w3 * inp
                + w2 * inp[0, 1, 0]
                + w1 * inp[0, 2, 0]
                + w0 * inp[0, 3, 0]
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
        inp: Field[float64],
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
            flux_x = flux_u(inp=inp, u=u, dx=dx)
            flux_y = flux_v(inp=inp, v=v, dy=dy)
            data_out = inp - dt * (  # noqa: data_out is modified to store the result
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
    def externals(cls):
        ext_dict = super().externals()
        ext_dict["K_OFFSET"] = 79
        return ext_dict

    @classmethod
    def stencil_definition(cls):
        """Return the stencil definition."""
        return cls._stencil_definition

    @staticmethod
    def _stencil_definition(
        out: Field[float64], inp: Field[float64], *, w: float64, dz: float64, dt: float64
    ):
        from __externals__ import K_OFFSET

        with computation(PARALLEL), interval(...):
            wf = w

        with computation(FORWARD):
            with interval(0, 1):
                a = -0.25 * wf / dz
                c = 0.25 * wf[0, 0, 1] / dz
                b = 1.0 / dt - a - c
                d = 1.0 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, K_OFFSET])
                alpha = -a
                gamma = -b
                b = 2 * b
                c = c / b
                d = d / b
                c2 = c / b
                d2 = gamma / b
            with interval(1, -1):
                a = -0.25 * wf / dz
                c = 0.25 * wf[0, 0, 1] / dz
                b = 1.0 / dt - a - c
                d = 1.0 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, -1])
                # alpha and gamma could be 2D
                alpha = alpha[0, 0, -1]
                gamma = gamma[0, 0, -1]
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                c2 = c / (b - c2[0, 0, -1] * a)
                d2 = (-a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

            with interval(-1, None):
                a = -0.25 * wf / dz
                c = 0.25 * wf[0, 0, -K_OFFSET] / dz
                b = 1.0 / dt - a - c
                d = 1.0 / dt * inp - c * (inp[0, 0, -K_OFFSET] - inp) + a * (inp - inp[0, 0, -1])
                # alpha and gamma could be 2D
                alpha = alpha[0, 0, -1]
                gamma = gamma[0, 0, -1]
                b = b + alpha * alpha / gamma
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                c2 = c / (b - c2[0, 0, -1] * a)
                d2 = (alpha - a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

        with computation(BACKWARD):
            with interval(0, 1):
                d = d - c * d[0, 0, 1]
                d2 = d2 - c2 * d2[0, 0, 1]
                fact = (d - alpha * d[0, 0, K_OFFSET] / gamma) / (
                    1 + d2 - alpha * d2[0, 0, K_OFFSET] / gamma
                )
            with interval(1, -1):
                d = d - c * d[0, 0, 1]
                d2 = d2 - c2 * d2[0, 0, 1]

        with computation(FORWARD):
            with interval(0, 1):
                out = d - fact * d2
            with interval(1, None):
                fact = fact[0, 0, -1]
                out = d - fact * d2  # noqa

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
    def externals(cls):
        ext_dict = super().externals()
        ext_dict["K_OFFSET"] = 79
        return ext_dict

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [Horizontal]

    def __call__(
        self, out: Field[float64], inp: Field[float64], inp0: Field[float64], *, dt: float64
    ) -> None:
        """Apply vertical advection stencil."""
        u, v, w = self.velocities
        dx, dy, dz = self.dspace
        self._call(out, inp, inp0, dx=dx, dy=dy, dz=dz, u=u, v=v, w=w, dt=dt)

    @classmethod
    def stencil_definition(cls):
        """Return the stencil definition."""
        return cls._stencil_definition

    @staticmethod
    def _stencil_definition(
        out: Field[float64],
        inp: Field[float64],
        inp0: Field[float64],
        *,
        u: float64,
        v: float64,
        w: float64,
        dx: float64,
        dy: float64,
        dz: float64,
        dt: float64,
    ):
        from __externals__ import K_OFFSET, flux_u, flux_v

        with computation(PARALLEL), interval(...):
            wf = w

        with computation(FORWARD):
            with interval(0, 1):
                a = -0.25 * wf / dz
                c = 0.25 * wf[0, 0, 1] / dz
                b = 1.0 / dt - a - c
                d = 1.0 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, K_OFFSET])
                alpha = -a
                gamma = -b
                b = 2 * b
                c = c / b
                d = d / b
                c2 = c / b
                d2 = gamma / b
            with interval(1, -1):
                a = -0.25 * wf / dz
                c = 0.25 * wf[0, 0, 1] / dz
                b = 1.0 / dt - a - c
                d = 1.0 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, -1])
                # alpha and gamma could be 2D
                alpha = alpha[0, 0, -1]
                gamma = gamma[0, 0, -1]
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                c2 = c / (b - c2[0, 0, -1] * a)
                d2 = (-a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

            with interval(-1, None):
                a = -0.25 * wf / dz
                c = 0.25 * wf[0, 0, -K_OFFSET] / dz
                b = 1.0 / dt - a - c
                d = 1.0 / dt * inp - c * (inp[0, 0, -K_OFFSET] - inp) + a * (inp - inp[0, 0, -1])
                # alpha and gamma could be 2D
                alpha = alpha[0, 0, -1]
                gamma = gamma[0, 0, -1]
                b = b + alpha * alpha / gamma
                c = c / (b - c[0, 0, -1] * a)
                d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                c2 = c / (b - c2[0, 0, -1] * a)
                d2 = (alpha - a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

        with computation(BACKWARD):
            with interval(0, 1):
                d = d - c * d[0, 0, 1]
                d2 = d2 - c2 * d2[0, 0, 1]
                fact = (d - alpha * d[0, 0, K_OFFSET] / gamma) / (
                    1 + d2 - alpha * d2[0, 0, K_OFFSET] / gamma
                )
            with interval(1, -1):
                d = d - c * d[0, 0, 1]
                d2 = d2 - c2 * d2[0, 0, 1]

        with computation(FORWARD):
            with interval(0, 1):
                vout = d - fact * d2
                flx = flux_u(inp=inp, u=u, dx=dx)
                fly = flux_v(inp=inp, v=v, dy=dy)
                out = inp0 - dt * (flx + fly) + (vout - inp)
            with interval(1, None):
                fact = fact[0, 0, -1]
                vout = d - fact * d2
                flx = flux_u(inp=inp, u=u, dx=dx)
                fly = flux_v(inp=inp, v=v, dy=dy)
                out = inp0 - dt * (flx + fly) + (vout - inp)  # noqa
