import click
import numpy as np

from ..verification import analytical
from ..utils.cli import cli_name, per_runtime_cli


def run_benchmark(runtime, domain_size, runs):
    nx, ny, nz = domain_size
    diffusion_coeff = 0.05
    exact = analytical.repeat(
        analytical.advection_diffusion(diffusion_coeff),
        ((nx + nz - 1) // nz, (ny + nz - 1) // nz, 1),
    )
    stepper = runtime.stencil_backend.advdiff_stepper(diffusion_coeff)

    lines = []

    def pfmt(*args):
        lines.append(args)

    print("Running GTBENCH")
    pfmt("Domain size:", f"{nx}x{ny}x{nz}")
    pfmt("Floating-point type:", runtime.stencil_backend.dtype)
    pfmt("GTBench4Py stencil backend:", cli_name(type(runtime.stencil_backend).__name__))
    pfmt("GTBench4Py runtime:", cli_name(type(runtime).__name__))

    results = [runtime.solve(exact, stepper, domain_size, 0.1, 1e-3) for _ in range(runs)]

    lower_time, median_time, upper_time = np.percentile([r.time for r in results], [2.5, 50, 97.5])
    conf = [f"(95% confidence: {lower_time}s - {upper_time}s)"] if runs > 100 else []
    pfmt("Median time:", f"{median_time}s", *conf)
    conf = (
        [f"(95% confidence: {nx * ny / upper_time}s - {nx * ny / lower_time}s)"]
        if runs > 100
        else []
    )
    pfmt("Columns per second:", nx * ny / median_time, *conf)

    align = 0
    for line in lines:
        align = max(align, len(str(line[0])))
    for line in lines:
        print(line[0].ljust(align), *line[1:])


cli = per_runtime_cli(
    run_benchmark,
    [
        click.option("--domain-size", type=int, nargs=3, required=True),
        click.option("--runs", default=101),
    ],
    dtype="float32",
)


if __name__ == "__main__":
    cli()
