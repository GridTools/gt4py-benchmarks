"""
Diffusion stencils.

Ported from GTBench by Rico Häuselmann <ricoh@cscs.ch>.
"""
from typing import Sequence

from numpy import float64
from gt4py.gtscript import Field, computation, interval, FORWARD, BACKWARD, PARALLEL

from gt4py_benchmarks.stencils.tooling import AbstractStencil


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
        out: Field[float64], inp: Field[float64], *, dz: float64, dt: float64, coeff: float64
    ):
        from __externals__ import K_OFFSET

        with computation(FORWARD):
            with interval(0, 1):
                ac = -coeff / (2.0 * dz * dz)
                b = 1.0 / dt - 2 * ac
                d = 1.0 / dt * inp + 0.5 * coeff * (
                    inp[0, 0, K_OFFSET] - 2 * inp + inp[0, 0, 1]
                ) / (dz * dz)
                b = 2 * b
                c = ac / b
                d = d / b
                c2 = c / b
                d2 = -0.5
            with interval(1, -1):
                ac = -coeff / (2 * dz * dz)
                b = 1.0 / dt - 2 * ac
                d = 1.0 / dt * inp + 0.5 * coeff * (inp[0, 0, -1] - 2 * inp + inp[0, 0, 1]) / (
                    dz * dz
                )
                c = ac / (b - c[0, 0, -1] * ac)
                d = (d - ac * d[0, 0, -1]) / (b - c[0, 0, -1] * ac)
                c2 = c / (b - c2[0, 0, -1] * ac)
                d2 = (-ac * d2[0, 0, -1]) / (b - c2[0, 0, -1] * ac)
            with interval(-1, None):
                ac = -coeff / (2 * dz * dz)
                b = 1.0 / dt - 2 * ac
                d = 1.0 / dt * inp + 0.5 * coeff * (
                    inp[0, 0, -1] - 2 * inp + inp[0, 0, -K_OFFSET]
                ) / (dz * dz)
                b = b + ac * ac / b
                c = ac / (b - c[0, 0, -1] * ac)
                d = (d - ac * d[0, 0, -1]) / (b - c[0, 0, -1] * ac)
                c2 = c / (b - c2[0, 0, -1] * ac)
                d2 = (ac - ac * d2[0, 0, -1]) / (b - c2[0, 0, -1] * ac)

        with computation(BACKWARD):
            with interval(1, -1):
                d = d - c * d[0, 0, 1]
                d2 = d2 - c2 * d2[0, 0, 1]
            with interval(0, 1):
                beta = -coeff / (2 * dz * dz)
                gamma = -1.0 / dt - 2 * beta
                d = d - c * d[0, 0, 1]
                d2 = d2 - c2 * d2[0, 0, 1]
                fact = (d + beta * d[0, 0, 79] / gamma) / (
                    1 + d2 + beta * d2[0, 0, K_OFFSET] / gamma
                )

        with computation(FORWARD):
            with interval(0, 1):
                out = d - fact * d2
            with interval(1, None):
                fact = fact[0, 0, -1]
                out = d - fact * d2  # noqa: F841

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
