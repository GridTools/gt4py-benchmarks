import importlib

import click
import numpy as np

from ..verification import analytical, convergence
from ..utils.cli import KeyValueArg


def run_convergence_tests(runtime):
    dtype = runtime.stencil_backend.dtype
    max_resolution = 16 if dtype == np.float32 else 32

    def run_tests(title, exact, stepper):
        print(f"=== {title} ===")
        print(f"Spatial convergence:")

        def spatial_error(n):
            return runtime.solve(
                exact, stepper, (n, n, n), 1e-2, 1e-3 if dtype == np.float32 else 1e-5
            ).error

        def spacetime_error(n):
            return runtime.solve(exact, stepper, (128, 128, 128), 1e-1, 1e-1 / n).error

        print(convergence.order_verification(spatial_error, 8, max_resolution))
        print(convergence.order_verification(spacetime_error, 8, max_resolution))

    diffusion_coeff = 0.05
    run_tests(
        "HORIZONTAL DIFFUSION",
        analytical.horizontal_diffusion(diffusion_coeff),
        runtime.stencil_backend.hdiff_stepper(diffusion_coeff),
    )
    run_tests(
        "VERTICAL DIFFUSION",
        analytical.vertical_diffusion(diffusion_coeff),
        runtime.stencil_backend.vdiff_stepper(diffusion_coeff),
    )
    run_tests(
        "FULL DIFFUSION",
        analytical.full_diffusion(diffusion_coeff),
        runtime.stencil_backend.diff_stepper(diffusion_coeff),
    )
    run_tests(
        "HORIZONTAL ADVECTION",
        analytical.horizontal_advection(),
        runtime.stencil_backend.hadv_stepper(),
    )
    run_tests(
        "VERTICAL ADVECTION",
        analytical.vertical_advection(),
        runtime.stencil_backend.vadv_stepper(),
    )


@click.command()
@click.option("--runtime", "-r", default="single_node")
@click.option("--stencil-backend", "-s", default="gt4py_backend")
@click.option("--dtype", default="float64")
@click.option("--stencil-backend-option", "-o", multiple=True, type=KeyValueArg())
def cli(runtime, stencil_backend, dtype, stencil_backend_option):
    stencil_backend = importlib.import_module(
        f"gt4py_benchmarks.numerics.stencil_backends.{stencil_backend}"
    ).StencilBackend(dtype=np.dtype(dtype), **dict(stencil_backend_option))
    runtime = importlib.import_module(f"gt4py_benchmarks.runtime.{runtime}").Runtime(
        stencil_backend
    )
    run_convergence_tests(runtime)
