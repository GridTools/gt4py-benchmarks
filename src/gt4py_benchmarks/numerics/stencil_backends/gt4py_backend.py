from gt4py import gtscript, storage
from gt4py.gtscript import Field, computation, interval, FORWARD, BACKWARD, PARALLEL

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


class StencilBackend(base.StencilBackend):
    def __init__(self, *, gt4py_backend="debug", **kwargs):
        super().__init__(**kwargs)
        self.gt4py_backend = gt4py_backend

    def storage_from_array(self, array):
        return storage.from_array(
            array,
            shape=array.shape,
            backend=self.gt4py_backend,
            default_origin=(HALO, HALO, 0),
            mask=(True, True, True),
        )

    def hdiff_stencil(self, resolution, delta, diffusion_coeff):
        @gtscript.stencil(backend=self.gt4py_backend)
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            dt: self.dtype.type,
            coeff: self.dtype.type,
            dx: self.dtype.type,
            dy: self.dtype.type,
        ):
            with computation(PARALLEL), interval(...):
                flx_x0 = _hdiff_limited_flux(
                    inp[-3, 0], inp[-2, 0], inp[-1, 0], inp, inp[1, 0], inp[2, 0], dx
                )
                flx_x1 = _hdiff_limited_flux(
                    inp[-2, 0], inp[-1, 0], inp, inp[1, 0], inp[2, 0], inp[3, 0], dx
                )
                flx_y0 = _hdiff_limited_flux(
                    inp[0, -3], inp[0, -2], inp[0, -1], inp, inp[0, 1], inp[0, 2], dy
                )
                flx_y1 = _hdiff_limited_flux(
                    inp[0, -2], inp[0, -1], inp, inp[0, 1], inp[0, 2], inp[0, 3], dy
                )

                out = inp + (
                    coeff * dt * (((flx_x1 - flx_x0) / dx) + ((flx_y1 - flx_y0) / dy))
                )  # noqa

        diffusion_coeff = self.dtype.type(diffusion_coeff)
        dx = self.dtype.type(delta[0])
        dy = self.dtype.type(delta[1])

        def wrapper(out, inp, dt):
            stencil(
                out,
                inp,
                self.dtype.type(dt),
                diffusion_coeff,
                dx,
                dy,
                origin=(HALO, HALO, 0),
                domain=resolution,
            )

        return wrapper

    def vdiff_stencil(self, resolution, delta, diffusion_coeff):
        @gtscript.stencil(
            backend=self.gt4py_backend, externals={"k_offset": int(resolution[2] - 1)}
        )
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            dt: self.dtype.type,
            coeff: self.dtype.type,
            dz: self.dtype.type,
        ):
            from __externals__ import k_offset

            with computation(FORWARD):
                with interval(0, 1):
                    ac = -coeff / (2.0 * dz * dz)
                    b = 1.0 / dt - 2 * ac
                    d = 1.0 / dt * inp + 0.5 * coeff * (
                        inp[0, 0, k_offset] - 2 * inp + inp[0, 0, 1]
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
                        inp[0, 0, -1] - 2 * inp + inp[0, 0, -k_offset]
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
                    fact = (d + beta * d[0, 0, k_offset] / gamma) / (
                        1 + d2 + beta * d2[0, 0, k_offset] / gamma
                    )

            with computation(FORWARD):
                with interval(0, 1):
                    out = d - fact * d2
                with interval(1, None):
                    fact = fact[0, 0, -1]
                    out = d - fact * d2  # noqa: F841

        diffusion_coeff = self.dtype.type(diffusion_coeff)
        dz = self.dtype.type(delta[2])

        def wrapper(out, inp, dt):
            stencil(
                out,
                inp,
                self.dtype.type(dt),
                diffusion_coeff,
                dz,
                origin=(HALO, HALO, 0),
                domain=resolution,
            )

        return wrapper

    def hadv_stencil(self, resolution, delta):
        @gtscript.stencil(backend=self.gt4py_backend)
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            u: Field[self.dtype.type],
            v: Field[self.dtype.type],
            dt: self.dtype.type,
            dx: self.dtype.type,
            dy: self.dtype.type,
        ):
            with computation(PARALLEL), interval(...):
                flux_x = _hadv_upwind_flux(
                    inp[-3, 0], inp[-2, 0], inp[-1, 0], inp, inp[1, 0], inp[2, 0], inp[3, 0], u, dx
                )
                flux_y = _hadv_upwind_flux(
                    inp[0, -3], inp[0, -2], inp[0, -1], inp, inp[0, 1], inp[0, 2], inp[0, 3], v, dy
                )
                out = inp - dt * (flux_x + flux_y)  # noqa

        dx = self.dtype.type(delta[0])
        dy = self.dtype.type(delta[1])

        def wrapper(out, inp, u, v, dt):
            stencil(
                out,
                inp,
                u,
                v,
                self.dtype.type(dt),
                dx,
                dy,
                origin=(HALO, HALO, 0),
                domain=resolution,
            )

        return wrapper

    def vadv_stencil(self, resolution, delta):
        @gtscript.stencil(
            backend=self.gt4py_backend, externals=dict(k_offset=int(resolution[2] - 1))
        )
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            w: Field[self.dtype.type],
            dt: self.dtype.type,
            dz: self.dtype.type,
        ):
            from __externals__ import k_offset

            with computation(FORWARD):
                with interval(0, 1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1.0 / dt - a - c
                    d = 1.0 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, k_offset])
                    alpha = -a
                    gamma = -b
                    b = 2 * b
                    c = c / b
                    d = d / b
                    c2 = c / b
                    d2 = gamma / b
                with interval(1, -1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
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
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, -k_offset] / dz
                    b = 1.0 / dt - a - c
                    d = (
                        1.0 / dt * inp
                        - c * (inp[0, 0, -k_offset] - inp)
                        + a * (inp - inp[0, 0, -1])
                    )
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
                    fact = (d - alpha * d[0, 0, k_offset] / gamma) / (
                        1 + d2 - alpha * d2[0, 0, k_offset] / gamma
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

        dz = self.dtype.type(delta[2])

        def wrapper(out, inp, w, dt):
            stencil(
                out, inp, w, self.dtype.type(dt), dz, origin=(HALO, HALO, 0), domain=resolution
            )

        return wrapper

    def rkadv_stencil(self, resolution, delta):
        @gtscript.stencil(
            backend=self.gt4py_backend, externals=dict(k_offset=int(resolution[2] - 1))
        )
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            inp0: Field[self.dtype.type],
            u: Field[self.dtype.type],
            v: Field[self.dtype.type],
            w: Field[self.dtype.type],
            dt: self.dtype.type,
            dx: self.dtype.type,
            dy: self.dtype.type,
            dz: self.dtype.type,
        ):
            from __externals__ import k_offset

            with computation(FORWARD):
                with interval(0, 1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
                    b = 1.0 / dt - a - c
                    d = 1.0 / dt * inp - c * (inp[0, 0, 1] - inp) + a * (inp - inp[0, 0, k_offset])
                    alpha = -a
                    gamma = -b
                    b = 2 * b
                    c = c / b
                    d = d / b
                    c2 = c / b
                    d2 = gamma / b
                with interval(1, -1):
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, 1] / dz
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
                    a = -0.25 * w / dz
                    c = 0.25 * w[0, 0, -k_offset] / dz
                    b = 1.0 / dt - a - c
                    d = (
                        1.0 / dt * inp
                        - c * (inp[0, 0, -k_offset] - inp)
                        + a * (inp - inp[0, 0, -1])
                    )
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
                    fact = (d - alpha * d[0, 0, k_offset] / gamma) / (
                        1 + d2 - alpha * d2[0, 0, k_offset] / gamma
                    )
                with interval(1, -1):
                    d = d - c * d[0, 0, 1]
                    d2 = d2 - c2 * d2[0, 0, 1]

            with computation(FORWARD):
                with interval(0, 1):
                    vout = d - fact * d2
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
                    out = inp0 - dt * (flx + fly) + (vout - inp)
                with interval(1, None):
                    fact = fact[0, 0, -1]
                    vout = d - fact * d2
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
                    out = inp0 - dt * (flx + fly) + (vout - inp)  # noqa

        dx = self.dtype.type(delta[0])
        dy = self.dtype.type(delta[1])
        dz = self.dtype.type(delta[2])

        def wrapper(out, inp, inp0, u, v, w, dt):
            stencil(
                out,
                inp,
                inp0,
                u,
                v,
                w,
                self.dtype.type(dt),
                dx,
                dy,
                dz,
                origin=(HALO, HALO, 0),
                domain=resolution,
            )

        return wrapper
