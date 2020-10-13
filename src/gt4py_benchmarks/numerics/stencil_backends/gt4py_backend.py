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

        @gtscript.stencil(
            backend=self.gt4py_backend,
            externals={
                "coeff": self.dtype.type(diffusion_coeff),
                "dx": self.dtype.type(delta[0]),
                "dy": self.dtype.type(delta[1]),
                "threshold": threshold,
            },
        )
        def stencil(out: Field[self.dtype.type], inp: Field[self.dtype.type], dt: self.dtype.type):

            from __externals__ import threshold, coeff, dx, dy  # noqa

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

        return stencil
