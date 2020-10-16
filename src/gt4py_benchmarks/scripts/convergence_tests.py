import importlib

import click
import numpy as np

from ..verification import analytical, convergence
from ..utils.cli import KeyValueArg


def run_convergence_tests(runtime):
    diffusion_coeff = 0.05
    print("=== HORIZONTAL DIFFUSION ===")
    print(
        convergence.default_convergence_test(
            runtime,
            analytical.horizontal_diffusion(diffusion_coeff),
            runtime.stencil_backend.hdiff_stepper(diffusion_coeff),
        ),
        end="",
    )
    print("=== VERTICAL DIFFUSION ===")
    print(
        convergence.default_convergence_test(
            runtime,
            analytical.vertical_diffusion(diffusion_coeff),
            runtime.stencil_backend.vdiff_stepper(diffusion_coeff),
        ),
        end="",
    )
    print("=== FULL DIFFUSION ===")
    print(
        convergence.default_convergence_test(
            runtime,
            analytical.full_diffusion(diffusion_coeff),
            runtime.stencil_backend.diff_stepper(diffusion_coeff),
        ),
        end="",
    )
    print("=== HORIZONTAL ADVECTION ===")
    print(
        convergence.default_convergence_test(
            runtime, analytical.horizontal_advection(), runtime.stencil_backend.hadv_stepper()
        ),
        end="",
    )
    print("=== VERTICAL ADVECTION ===")
    print(
        convergence.default_convergence_test(
            runtime, analytical.vertical_advection(), runtime.stencil_backend.vadv_stepper()
        ),
        end="",
    )
    print("=== RUNGE-KUTTA ADVECTION ===")
    print(
        convergence.default_convergence_test(
            runtime, analytical.full_advection(), runtime.stencil_backend.rkadv_stepper()
        ),
        end="",
    )
    print("=== ADVECTION-DIFFUSION ===")
    print(
        convergence.default_convergence_test(
            runtime,
            analytical.advection_diffusion(diffusion_coeff),
            runtime.stencil_backend.advdiff_stepper(diffusion_coeff),
        ),
        end="",
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
