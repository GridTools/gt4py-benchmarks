import click

from ..verification import convergence
from ..utils.cli import per_runtime_cli


def run_convergence_tests(runtime, mode):
    for test in convergence.default_convergence_tests(runtime, full_range=mode == "full-range"):
        result = test()
        print(result, end="")


cli = per_runtime_cli(
    run_convergence_tests,
    [click.option("--mode", type=click.Choice(["fast", "full-range"]), default="fast")],
    dtype="float64",
)


if __name__ == "__main__":
    cli()
