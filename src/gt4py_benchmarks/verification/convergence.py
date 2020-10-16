import typing

import numpy as np


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
    spatial: OrderVerificationResult
    temporal: OrderVerificationResult

    def __str__(self):
        return f"Spatial convergence:\n{self.spatial}Temporal convergence:\n{self.temporal}"


def default_convergence_test(runtime, analytical, stepper):
    dtype = np.dtype(runtime.stencil_backend.dtype)

    def spatial_error(n):
        tmax = 2e-2 if dtype == np.float32 else 1e-3
        return runtime.solve(analytical, stepper, (n, n, n), tmax, tmax / 100).error

    def temporal_error(n):
        tmax = 1e-1 if dtype == np.float32 else 1e-2
        return runtime.solve(analytical, stepper, (128, 128, 128), tmax, tmax / n).error

    n = 16 if dtype == np.float32 else 32
    return ConvergenceTestResult(
        spatial=order_verification(spatial_error, n // 2, n),
        temporal=order_verification(temporal_error, 8, 16),
    )
