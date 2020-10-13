from gt4py import gtscript, storage
from gt4py.gtscript import Field, computation, interval, FORWARD, BACKWARD, PARALLEL

from . import base
from ...constants import HALO


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
        @gtscript.function
        def threshold(val, diff):
            return 0 if val * diff < 0 else val

        @gtscript.stencil(backend=self.gt4py_backend, externals={"threshold": threshold})
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            dt: self.dtype.type,
            coeff: self.dtype.type,
            dx: self.dtype.type,
            dy: self.dtype.type,
        ):

            from __externals__ import threshold  # noqa

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
            backend=self.gt4py_backend, externals={"K_OFFSET": int(resolution[2] - 1)}
        )
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            dt: self.dtype.type,
            coeff: self.dtype.type,
            dz: self.dtype.type,
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
                    fact = (d + beta * d[0, 0, K_OFFSET] / gamma) / (
                        1 + d2 + beta * d2[0, 0, K_OFFSET] / gamma
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
        @gtscript.function
        def flux(im3, im2, im1, ic, ip1, ip2, ip3, velocity, delta):
            from __externals__ import w0, w1, w2, w3, w4, w5

            if_pos = -(
                velocity * (w0 * im3 + w1 * im2 + w2 * im1 + w3 * ic + w4 * ip1 + w5 * ip2) / delta
            )
            if_neg = (
                velocity * (w5 * im2 + w4 * im1 + w3 * ic + w2 * ip1 + w1 * ip2 + w0 * ip3) / delta
            )
            return if_pos if velocity > 0 else if_neg

        weights = 1 / 30, -1 / 4, 1, -1 / 3, -1 / 2, 1 / 20
        externals = {f"w{i}": self.dtype.type(w) for i, w in enumerate(weights)}
        externals["flux"] = flux

        @gtscript.stencil(backend=self.gt4py_backend, externals=externals)
        def stencil(
            out: Field[self.dtype.type],
            inp: Field[self.dtype.type],
            u: Field[self.dtype.type],
            v: Field[self.dtype.type],
            dt: self.dtype.type,
            dx: self.dtype.type,
            dy: self.dtype.type,
        ):
            from __externals__ import flux  # noqa

            with computation(PARALLEL), interval(...):
                flux_x = flux(
                    inp[-3, 0], inp[-2, 0], inp[-1, 0], inp, inp[1, 0], inp[2, 0], inp[3, 0], u, dx
                )
                flux_y = flux(
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
