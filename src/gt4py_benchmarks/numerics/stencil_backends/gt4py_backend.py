import gt4py
from gt4py import backend, gtscript, storage
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

    def _modified_gt4py_backend(self, resolution):
        if "dace" not in self.gt4py_backend:
            return self.gt4py_backend
        backend = gt4py.backend.base.from_name(self.gt4py_backend)
        new_name = self.gt4py_backend + "_" + "x".join(str(r) for r in resolution)
        assert backend.storage_info["layout_map"]([True] * 3) == (2, 1, 0)
        strides = (
            1,
            resolution[0] + 2 * HALO,
            (resolution[0] + 2 * HALO) * (resolution[1] + 2 * HALO),
        )

        class Optimizer(type(backend.DEFAULT_OPTIMIZER)):
            def transform_optimize(self, sdfg):
                symbols = {d: np.int32(r) for d, r in zip("IJK", resolution)}
                for name in sdfg.arrays:
                    symbols.update(
                        {f"_{name}_{d}_stride": np.int32(s) for d, s in zip("IJK", strides)}
                    )
                sdfg.specialize(symbols)
                for nsdfg in sdfg.all_sdfgs_recursive():
                    nsdfg.specialize(symbols)
                return super().transform_optimize(sdfg)

        class Backend(backend):
            name = new_name
            DEFAULT_OPTIMIZER = Optimizer()

        try:
            gt4py.backend.base.register(Backend)
        except ValueError:
            pass

        return new_name

    def storage_from_array(self, array):
        return storage.from_array(
            array,
            shape=array.shape,
            backend=self.gt4py_backend,
            default_origin=(HALO, HALO, 0),
            mask=(True, True, True),
            managed_memory=True,
        )

    def synchronize(self):
        if self.gt4py_backend == "dacecuda":
            import cupy as cp

            cp.cuda.Device().synchronize()

    def hdiff_stencil(self, resolution, delta, diffusion_coeff):
        @gtscript.stencil(
            backend=self._modified_gt4py_backend(resolution), enforce_dtype=self.dtype
        )
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
            validate_args=False,
        )

    def vdiff_stencil(self, resolution, delta, diffusion_coeff):
        @gtscript.stencil(
            backend=self._modified_gt4py_backend(resolution),
            enforce_dtype=self.dtype,
            externals=dict(k_offset=int(resolution[2] - 1)),
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
                    a = -coeff / (2 * dz * dz)
                    b = 1 / dt + coeff / (dz * dz)
                    d1 = 1 / dt * inp + 0.5 * coeff * (
                        inp[0, 0, k_offset] - 2 * inp + inp[0, 0, 1]
                    ) / (dz * dz)
                    d2 = -a
                with interval(1, -2):
                    a = -coeff / (2 * dz * dz)
                    b = 1 / dt + coeff / (dz * dz)
                    c = a
                    d1 = 1 / dt * inp + 0.5 * coeff * (inp[0, 0, -1] - 2 * inp + inp[0, 0, 1]) / (
                        dz * dz
                    )
                    d2 = 0

                    f = a / b[0, 0, -1]
                    b -= f * c
                    d1 -= f * d1[0, 0, -1]
                    d2 -= f * d2[0, 0, -1]
                with interval(-2, -1):
                    a = -coeff / (2 * dz * dz)
                    b = 1 / dt + coeff / (dz * dz)
                    c = a
                    d1 = 1 / dt * inp + 0.5 * coeff * (inp[0, 0, -1] - 2 * inp + inp[0, 0, 1]) / (
                        dz * dz
                    )
                    d2 = -c

                    f = a / b[0, 0, -1]
                    b -= f * c
                    d1 -= f * d1[0, 0, -1]
                    d2 -= f * d2[0, 0, -1]

            with computation(BACKWARD):
                with interval(-2, -1):
                    f = 1 / b
                    d1 *= f
                    d2 *= f
                with interval(0, -2):
                    c = -coeff / (2 * dz * dz)
                    f = 1 / b
                    d1 = (d1 - c * d1[0, 0, 1]) * f
                    d2 = (d2 - c * d2[0, 0, 1]) * f

            # workaround for https://github.com/GridTools/gt4py/issues/246
            with computation(FORWARD):
                with interval(-1, None):
                    a = -coeff / (2 * dz * dz)
                    b = 1 / dt + coeff / (dz * dz)
                    c = a
                    d1 = 1 / dt * inp + 0.5 * coeff * (
                        inp[0, 0, -1] - 2 * inp + inp[0, 0, -k_offset]
                    ) / (dz * dz)

            with computation(BACKWARD):
                with interval(-1, None):
                    # a = -coeff / (2 * dz * dz)
                    # b = 1 / dt + coeff / (dz * dz)
                    # c = a
                    # d1 = 1 / dt * inp + 0.5 * coeff * (
                    # inp[0, 0, -1] - 2 * inp + inp[0, 0, -k_offset]
                    # ) / (dz * dz)

                    out_top = (d1 - c * d1[0, 0, -k_offset] - a * d1[0, 0, -1]) / (
                        b + c * d2[0, 0, -k_offset] + a * d2[0, 0, -1]
                    )
                    out = out_top
                with interval(0, -1):
                    out_top = out_top[0, 0, 1]
                    out = d1 + d2 * out_top

        return lambda out, inp, dt: vdiff(
            out,
            inp,
            self._scalar(dt),
            self._scalar(diffusion_coeff),
            self._scalar(delta[2]),
            domain=resolution,
            validate_args=False,
        )

    def hadv_stencil(self, resolution, delta):
        @gtscript.stencil(
            backend=self._modified_gt4py_backend(resolution), enforce_dtype=self.dtype
        )
        def hadv(
            out: self._field,
            inp: self._field,
            inp0: self._field,
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
                out = inp0 - dt * (flux_x + flux_y)  # noqa

        return lambda out, inp, inp0, u, v, dt: hadv(
            out,
            inp,
            inp0,
            u,
            v,
            self._scalar(dt),
            self._scalar(delta[0]),
            self._scalar(delta[1]),
            domain=resolution,
            validate_args=False,
        )

    def vadv_stencil(self, resolution, delta):
        @gtscript.stencil(
            backend=self._modified_gt4py_backend(resolution),
            enforce_dtype=self.dtype,
            externals=dict(k_offset=int(resolution[2] - 1)),
        )
        def vadv(
            out: self._field,
            inp: self._field,
            inp0: self._field,
            w: self._field,
            dt: self._scalar,
            dz: self._scalar,
        ):
            from __externals__ import k_offset

            with computation(FORWARD):
                with interval(0, 1):
                    a = -0.25 / dz * w
                    b = 1 / dt + 0.25 * (w - w[0, 0, 1]) / dz
                    d1 = 1 / dt * inp - 0.25 / dz * (
                        w * (inp - inp[0, 0, k_offset]) + w[0, 0, 1] * (inp[0, 0, 1] - inp)
                    )
                    d2 = -a
                with interval(1, -2):
                    a = -0.25 / dz * w
                    b = 1 / dt + 0.25 * (w - w[0, 0, 1]) / dz
                    c_km1 = -a
                    d1 = 1 / dt * inp - 0.25 / dz * (
                        w * (inp - inp[0, 0, -1]) + w[0, 0, 1] * (inp[0, 0, 1] - inp)
                    )
                    d2 = 0

                    f = a / b[0, 0, -1]
                    b -= f * c_km1
                    d1 -= f * d1[0, 0, -1]
                    d2 -= f * d2[0, 0, -1]
                with interval(-2, -1):
                    a = -0.25 / dz * w
                    b = 1 / dt + 0.25 * (w - w[0, 0, 1]) / dz
                    c = 0.25 / dz * w[0, 0, 1]
                    c_km1 = -a
                    d1 = 1 / dt * inp - 0.25 / dz * (
                        w * (inp - inp[0, 0, -1]) + w[0, 0, 1] * (inp[0, 0, 1] - inp)
                    )
                    d2 = -c

                    f = a / b[0, 0, -1]
                    b -= f * c_km1
                    d1 -= f * d1[0, 0, -1]
                    d2 -= f * d2[0, 0, -1]

            with computation(BACKWARD):
                with interval(-2, -1):
                    f = 1 / b
                    d1 *= f
                    d2 *= f
                with interval(0, -2):
                    c = 0.25 / dz * w[0, 0, 1]
                    f = 1 / b
                    d1 = (d1 - c * d1[0, 0, 1]) * f
                    d2 = (d2 - c * d2[0, 0, 1]) * f

            # workaround for https://github.com/GridTools/gt4py/issues/246
            with computation(FORWARD):
                with interval(-1, None):
                    a = -0.25 / dz * w
                    # in C++ we use w[0, 0, 1] instead of w[0, 0, -k_offset]
                    b = 1 / dt + 0.25 * (w - w[0, 0, -k_offset]) / dz
                    c = 0.25 / dz * w[0, 0, -k_offset]
                    d1 = 1 / dt * inp - 0.25 / dz * (
                        w * (inp - inp[0, 0, -1])
                        + w[0, 0, -k_offset] * (inp[0, 0, -k_offset] - inp)
                    )

            with computation(BACKWARD):
                with interval(-1, None):
                    # a = -0.25 / dz * w
                    # b = 1 / dt + 0.25 * (w - w[0, 0, 1]) / dz
                    # c = 0.25 / dz * w[0, 0, 1]
                    # d1 = 1 / dt * inp - 0.25 / dz * (
                    # w * (inp - inp[0, 0, -1]) + w[0, 0, 1] * (inp[0, 0, -k_offset] - inp)
                    # )

                    out_top = (d1 - c * d1[0, 0, -k_offset] - a * d1[0, 0, -1]) / (
                        b + c * d2[0, 0, -k_offset] + a * d2[0, 0, -1]
                    )
                    out = inp0 + (out_top - inp)
                with interval(0, -1):
                    out_top = out_top[0, 0, 1]
                    out = inp0 + (d1 + d2 * out_top - inp)

        return lambda out, inp, inp0, w, dt: vadv(
            out,
            inp,
            inp0,
            w,
            self._scalar(dt),
            self._scalar(delta[2]),
            domain=resolution,
            validate_args=False,
        )
