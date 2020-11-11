import itertools

import click

from ..verification import convergence
from ..utils.cli import per_runtime_cli


def run_convergence_tests(runtime):
    for test in convergence.default_convergence_tests(runtime):
        result = test()
        print(result, end="")

cli = per_runtime_cli(run_convergence_tests, dtype="float64")


if __name__ == "__main__":
    cli()
