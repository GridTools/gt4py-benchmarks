import itertools

import click

from ..verification import analytical, convergence
from ..utils.cli import per_runtime_cli


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


cli = per_runtime_cli(run_convergence_tests, dtype="float64")


if __name__ == "__main__":
    cli()
