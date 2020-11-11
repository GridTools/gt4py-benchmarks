import typing

import numpy as np

from . import analytical


class OrderVerificationResult(typing.NamedTuple):
    ns: np.ndarray
    errors: np.ndarray
    orders: np.ndarray

    def __str__(self):
        s = "{:15} {:15} {:15}\n".format("Resolution", "Error", "Order")
        for n, e, o in zip(self.ns, self.errors, self.orders):
            s += f"{n:<15} {e:<15.5e}"
            if not np.isnan(o):
                s += f" {o:<15.2f}"
            s += "\n"
        return s


def order_verification(f, n_min, n_max):
    ns = n_min * 2 ** np.arange(int(np.log2(n_max / n_min)) + 1)
    errors = np.array([f(n) for n in ns])
    orders = np.empty_like(errors)
    orders[0] = np.nan
    orders[1:] = np.log2(errors[:-1] / errors[1:])

    return OrderVerificationResult(ns, errors, orders)


class ConvergenceTestResult(typing.NamedTuple):
    name: str
    spatial: OrderVerificationResult
    temporal: OrderVerificationResult

    def __str__(self):
        return (
            f"=== {self.name.upper()} ===\n"
            f"Spatial convergence:\n{self.spatial}"
            f"Temporal convergence:\n{self.temporal}"
        )


def convergence_test(
    name,
    runtime,
    analytical,
    stepper,
    tmax_spatial,
    n_spatial,
    tmax_temporal,
    n_temporal,
    full_range=False,
):
    dtype = np.dtype(runtime.stencil_backend.dtype)

    def spatial_error(n):
        return runtime.solve(
            analytical, stepper, (n, n, n), tmax_spatial, tmax_spatial / 100
        ).error

    def temporal_error(n):
        return runtime.solve(
            analytical, stepper, (32, 32, 1024), tmax_temporal, tmax_temporal / n
        ).error

    if full_range:
        nmin_spatial = nmin_temporal = 2
        nmax_spatial = nmax_temporal = 128
    else:
        nmin_spatial = n_spatial // 2
        nmax_spatial = n_spatial
        nmin_temporal = n_temporal // 2
        nmax_temporal = n_temporal

    def run():
        return ConvergenceTestResult(
            name=name,
            spatial=order_verification(spatial_error, nmin_spatial, nmax_spatial),
            temporal=order_verification(temporal_error, nmin_temporal, nmax_temporal),
        )

    return run


class DefaultConvergenceTests(typing.NamedTuple):
    hdiff: typing.Callable[[], ConvergenceTestResult]
    vdiff: typing.Callable[[], ConvergenceTestResult]
    diff: typing.Callable[[], ConvergenceTestResult]
    hadv: typing.Callable[[], ConvergenceTestResult]
    vadv: typing.Callable[[], ConvergenceTestResult]
    rkadv: typing.Callable[[], ConvergenceTestResult]
    advdiff: typing.Callable[[], ConvergenceTestResult]


def default_convergence_tests(runtime):
    diffusion_coeff = 0.05
    is_float = runtime.stencil_backend.dtype == "float32"
    return DefaultConvergenceTests(
        hdiff=convergence_test(
            "horizontal diffusion",
            runtime,
            analytical.horizontal_diffusion(diffusion_coeff),
            runtime.stencil_backend.hdiff_stepper(diffusion_coeff),
            1e-1 if is_float else 1e-3,
            16 if is_float else 32,
            5e-1,
            16,
        ),
        vdiff=convergence_test(
            "vertical diffusion",
            runtime,
            analytical.vertical_diffusion(diffusion_coeff),
            runtime.stencil_backend.vdiff_stepper(diffusion_coeff),
            5,
            64,
            50,
            8 if is_float else 16,
        ),
        diff=convergence_test(
            "full diffusion",
            runtime,
            analytical.full_diffusion(diffusion_coeff),
            runtime.stencil_backend.diff_stepper(diffusion_coeff),
            1e-1 if is_float else 1e-3,
            32,
            5e-1,
            16,
        ),
        hadv=convergence_test(
            "horizontal advection",
            runtime,
            analytical.horizontal_advection(),
            runtime.stencil_backend.hadv_stepper(),
            1e-1 if is_float else 1e-4,
            32 if is_float else 64,
            1e-1,
            16,
        ),
        vadv=convergence_test(
            "vertical advection",
            runtime,
            analytical.vertical_advection(),
            runtime.stencil_backend.vadv_stepper(),
            1e-1,
            128,
            10,
            32,
        ),
        rkadv=convergence_test(
            "runge-kutta advection",
            runtime,
            analytical.full_advection(),
            runtime.stencil_backend.rkadv_stepper(),
            1e-2,
            64,
            1,
            8,
        ),
        advdiff=convergence_test(
            "advection-diffusion",
            runtime,
            analytical.advection_diffusion(diffusion_coeff),
            runtime.stencil_backend.advdiff_stepper(diffusion_coeff),
            1e-1 if is_float else 1e-3,
            64,
            1,
            16 if is_float else 64,
        ),
    )
