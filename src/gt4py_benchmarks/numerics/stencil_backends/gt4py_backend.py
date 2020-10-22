from gt4py import gtscript, storage
from gt4py.gtscript import Field, computation, interval, FORWARD, BACKWARD, PARALLEL
import numpy as np
import typing_extensions

from . import base
from ...constants import HALO


@gtscript.function
def _hdiff_flux(im25, im15, im05, ip05, ip15, ip25, delta):
    return (
        -im25 * (1.0 / 90)
        + im15 * (5.0 / 36.0)
        - im05 * (49.0 / 36.0)
        + ip05 * (49.0 / 36.0)
        - ip15 * (5.0 / 36.0)
        + ip25 * (1.0 / 90.0)
    ) / delta


@gtscript.function
def _hdiff_limited_flux(im25, im15, im05, ip05, ip15, ip25, delta):
    flux = _hdiff_flux(im25, im15, im05, ip05, ip15, ip25, delta)
    return 0 if flux * (ip05 - im05) < 0 else flux


@gtscript.function
def _hadv_flux(im25, im15, im05, ip05, ip15, ip25, velocity, delta):
    return (
        -velocity
        * (
            im25 * (1.0 / 30.0)
            - im15 * (1.0 / 4.0)
            + im05
            - ip05 * (1.0 / 3.0)
            - ip15 * (1.0 / 2.0)
            + ip25 * (1.0 / 20.0)
        )
        / delta
    )


@gtscript.function
def _hadv_upwind_flux(im3, im2, im1, ic, ip1, ip2, ip3, velocity, delta):
    return (
        _hadv_flux(im3, im2, im1, ic, ip1, ip2, velocity, delta)
        if velocity > 0
        else _hadv_flux(ip3, ip2, ip1, ic, im1, im2, -velocity, delta)
    )


class GT4PyStencilBackend(base.StencilBackend):
    gt4py_backend: typing_extensions.Literal[
        "debug", "numpy", "gtx86", "gtmc", "gtcuda", "dacex86", "dacecuda"
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._field = Field[np.dtype(self.dtype).type]
        self._scalar = np.dtype(self.dtype).type

    def storage_from_array(self, array):
        return storage.from_array(
            array,
            shape=array.shape,
            backend=self.gt4py_backend,
            default_origin=(HALO, HALO, 0),
            mask=(True, True, True),
            managed_memory=True,
        )

    def hdiff_stencil(self, resolution, delta, diffusion_coeff):
        @gtscript.stencil(backend=self.gt4py_backend)
        def hdiff(
            out: self._field,
            inp: self._field,
            dt: self._scalar,
            coeff: self._scalar,
            dx: self._scalar,
            dy: self._scalar,
        ):
            with computation(PARALLEL), interval(...):
                flx = _hdiff_limited_flux(
                    inp[-3, 0], inp[-2, 0], inp[-1, 0], inp, inp[1, 0], inp[2, 0], dx
                )
                fly = _hdiff_limited_flux(
                    inp[0, -3], inp[0, -2], inp[0, -1], inp, inp[0, 1], inp[0, 2], dy
                )
                out = inp + coeff * dt * ((flx[1, 0] - flx) / dx + (fly[0, 1] - fly) / dy)

        return lambda out, inp, dt: hdiff(
            out,
            inp,
            self._scalar(dt),
            self._scalar(diffusion_coeff),
            self._scalar(delta[0]),
            self._scalar(delta[1]),
            domain=resolution,
        )

    def vdiff_stencil(self, resolution, delta, diffusion_coeff):
        @gtscript.stencil(
            backend=self.gt4py_backend, externals=dict(k_offset=int(resolution[2] - 1))
        )
        def vdiff(
            out: self._field,
            inp: self._field,
            dt: self._scalar,
            coeff: self._scalar,
            dz: self._scalar,
        ):
            from __externals__ import k_offset

            with computation(FORWARD):
                with interval(0, 1):
                    ac = -coeff / (2 * dz * dz)
                    b = 1 / dt - 2 * ac
                    d = 1 / dt * inp + 0.5 * coeff * (
                        inp[0, 0, k_offset] - 2 * inp + inp[0, 0, 1]
                    ) / (dz * dz)
                    b = 2 * b
                    c = ac / b
                    d = d / b
                    c2 = c / b
                    d2 = -0.5
                with interval(1, -1):
                    ac = -coeff / (2 * dz * dz)
                    b = 1 / dt - 2 * ac
                    d = 1 / dt * inp + 0.5 * coeff * (inp[0, 0, -1] - 2 * inp + inp[0, 0, 1]) / (
                        dz * dz
                    )
                    c = ac / (b - c[0, 0, -1] * ac)
                    d = (d - ac * d[0, 0, -1]) / (b - c[0, 0, -1] * ac)
                    c2 = c / (b - c2[0, 0, -1] * ac)
                    d2 = (-ac * d2[0, 0, -1]) / (b - c2[0, 0, -1] * ac)
                with interval(-1, None):
                    ac = -coeff / (2 * dz * dz)
                    b = 1 / dt - 2 * ac
                    d = 1 / dt * inp + 0.5 * coeff * (
                        inp[0, 0, -1] - 2 * inp + inp[0, 0, -k_offset]
                    ) / (dz * dz)
                    b = b + ac * ac / b
                    c = ac / (b - c[0, 0, -1] * ac)
                    d = (d - ac * d[0, 0, -1]) / (b - c[0, 0, -1] * ac)
                    c2 = c / (b - c2[0, 0, -1] * ac)
                    d2 = (ac - ac * d2[0, 0, -1]) / (b - c2[0, 0, -1] * ac)

            with computation(BACKWARD):
                with interval(0, -1):
                    d = d - c * d[0, 0, 1]
                    d2 = d2 - c2 * d2[0, 0, 1]

            with computation(FORWARD):
                with interval(0, 1):
                    ac = -coeff / (2 * dz * dz)
                    b = -(1 / dt - 2 * ac)
                    fact = (d + ac * d[0, 0, k_offset] / b) / (
                        1 + d2 + ac * d2[0, 0, k_offset] / b
                    )
                    out = d - fact * d2
                with interval(1, None):
                    fact = fact[0, 0, -1]
                    out = d - fact * d2  # noqa: F841

        return lambda out, inp, dt: vdiff(
            out,
            inp,
            self._scalar(dt),
            self._scalar(diffusion_coeff),
            self._scalar(delta[2]),
            domain=resolution,
        )

    def hadv_stencil(self, resolution, delta):
        @gtscript.stencil(backend=self.gt4py_backend)
        def hadv(
            out: self._field,
            inp: self._field,
            u: self._field,
            v: self._field,
            dt: self._scalar,
            dx: self._scalar,
            dy: self._scalar,
        ):
            with computation(PARALLEL), interval(...):
                flux_x = _hadv_upwind_flux(
                    inp[-3, 0], inp[-2, 0], inp[-1, 0], inp, inp[1, 0], inp[2, 0], inp[3, 0], u, dx
                )
                flux_y = _hadv_upwind_flux(
                    inp[0, -3], inp[0, -2], inp[0, -1], inp, inp[0, 1], inp[0, 2], inp[0, 3], v, dy
                )
                out = inp - dt * (flux_x + flux_y)  # noqa

        return lambda out, inp, u, v, dt: hadv(
            out,
            inp,
            u,
            v,
            self._scalar(dt),
            self._scalar(delta[0]),
            self._scalar(delta[1]),
            domain=resolution,
        )

    def vadv_stencil(self, resolution, delta):
        @gtscript.stencil(
            backend=self.gt4py_backend, externals=dict(k_offset=int(resolution[2] - 1))
        )
        def vadv(
            out: self._field, inp: self._field, w: self._field, dt: self._scalar, dz: self._scalar
        ):
            from __externals__ import k_offset

            with computation(FORWARD):
                with interval(0, 1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1 / dt - a - c
                    d = 1 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, k_offset])
                    a0 = a
                    b0 = b
                    b = 2 * b
                    c = c / b
                    d = d / b
                    c2 = c / b
                    d2 = -b0 / b
                with interval(1, -1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1 / dt - a - c
                    d = 1 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, -1])
                    # a0 and b0 could be 2D
                    a0 = a0[0, 0, -1]
                    b0 = b0[0, 0, -1]
                    c = c / (b - c[0, 0, -1] * a)
                    d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                    c2 = c / (b - c2[0, 0, -1] * a)
                    d2 = (-a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

                with interval(-1, None):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, -k_offset] / dz
                    b = 1 / dt - a - c
                    d = 1 / dt * inp - c * (inp[0, 0, -k_offset] - inp) + a * (inp - inp[0, 0, -1])
                    # a0 and b0 could be 2D
                    a0 = a0[0, 0, -1]
                    b0 = b0[0, 0, -1]
                    b = b - a0 * a0 / b0
                    c = c / (b - c[0, 0, -1] * a)
                    d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                    c2 = c / (b - c2[0, 0, -1] * a)
                    d2 = (-a0 - a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

            with computation(BACKWARD):
                with interval(0, -1):
                    d = d - c * d[0, 0, 1]
                    d2 = d2 - c2 * d2[0, 0, 1]

            with computation(FORWARD):
                with interval(0, 1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1 / dt - a - c
                    fact = (d - a * d[0, 0, k_offset] / b) / (1 + d2 - a * d2[0, 0, k_offset] / b)
                    out = d - fact * d2
                with interval(1, None):
                    fact = fact[0, 0, -1]
                    out = d - fact * d2  # noqa

        return lambda out, inp, w, dt: vadv(
            out, inp, w, self._scalar(dt), self._scalar(delta[2]), domain=resolution
        )

    def rkadv_stencil(self, resolution, delta):
        @gtscript.stencil(
            backend=self.gt4py_backend, externals=dict(k_offset=int(resolution[2] - 1))
        )
        def rkadv(
            out: self._field,
            inp: self._field,
            inp0: self._field,
            u: self._field,
            v: self._field,
            w: self._field,
            dt: self._scalar,
            dx: self._scalar,
            dy: self._scalar,
            dz: self._scalar,
        ):
            from __externals__ import k_offset

            with computation(FORWARD):
                with interval(0, 1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1 / dt - a - c
                    d = 1 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, k_offset])
                    a0 = a
                    b0 = b
                    b = 2 * b
                    c = c / b
                    d = d / b
                    c2 = c / b
                    d2 = -b0 / b
                with interval(1, -1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1 / dt - a - c
                    d = 1 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, -1])
                    # a0 and b0 could be 2D
                    a0 = a0[0, 0, -1]
                    b0 = b0[0, 0, -1]
                    c = c / (b - c[0, 0, -1] * a)
                    d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                    c2 = c / (b - c2[0, 0, -1] * a)
                    d2 = (-a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

                with interval(-1, None):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, -k_offset] / dz
                    b = 1 / dt - a - c
                    d = 1 / dt * inp - c * (inp[0, 0, -k_offset] - inp) + a * (inp - inp[0, 0, -1])
                    # a0 and b0 could be 2D
                    a0 = a0[0, 0, -1]
                    b0 = b0[0, 0, -1]
                    b = b - a0 * a0 / b0
                    c = c / (b - c[0, 0, -1] * a)
                    d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
                    c2 = c / (b - c2[0, 0, -1] * a)
                    d2 = (-a0 - a * d2[0, 0, -1]) / (b - c2[0, 0, -1] * a)

            with computation(BACKWARD):
                with interval(0, -1):
                    d = d - c * d[0, 0, 1]
                    d2 = d2 - c2 * d2[0, 0, 1]

            with computation(FORWARD):
                with interval(0, 1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1 / dt - a - c
                    fact = (d - a * d[0, 0, k_offset] / b) / (1 + d2 - a * d2[0, 0, k_offset] / b)
                    flx = _hadv_upwind_flux(
                        inp[-3, 0],
                        inp[-2, 0],
                        inp[-1, 0],
                        inp,
                        inp[1, 0],
                        inp[2, 0],
                        inp[3, 0],
                        u,
                        dx,
                    )
                    fly = _hadv_upwind_flux(
                        inp[0, -3],
                        inp[0, -2],
                        inp[0, -1],
                        inp,
                        inp[0, 1],
                        inp[0, 2],
                        inp[0, 3],
                        v,
                        dy,
                    )
                    vout = d - fact * d2
                    out = inp0 - dt * (flx + fly) + (vout - inp)
                with interval(1, None):
                    fact = fact[0, 0, -1]
                    flx = _hadv_upwind_flux(
                        inp[-3, 0],
                        inp[-2, 0],
                        inp[-1, 0],
                        inp,
                        inp[1, 0],
                        inp[2, 0],
                        inp[3, 0],
                        u,
                        dx,
                    )
                    fly = _hadv_upwind_flux(
                        inp[0, -3],
                        inp[0, -2],
                        inp[0, -1],
                        inp,
                        inp[0, 1],
                        inp[0, 2],
                        inp[0, 3],
                        v,
                        dy,
                    )
                    vout = d - fact * d2
                    out = inp0 - dt * (flx + fly) + (vout - inp)  # noqa

        return lambda out, inp, inp0, u, v, w, dt: rkadv(
            out,
            inp,
            inp0,
            u,
            v,
            w,
            self._scalar(dt),
            self._scalar(delta[0]),
            self._scalar(delta[1]),
            self._scalar(delta[2]),
            domain=resolution,
        )
