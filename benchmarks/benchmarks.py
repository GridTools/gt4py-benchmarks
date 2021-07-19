"""GT4Py benchmarking suite."""
import sys
import pathlib

bpath = pathlib.Path(__file__).parent
sys.path.append(str(bpath.parent / "src"))

from gt4py_benchmarks.runtime.runtimes.single_node import (  # noqa (path appending required first)
    SingleNodeRuntime,  # noqa (path appending required first)
)  # noqa (path appending required first)
from gt4py_benchmarks.numerics.stencil_backends.gt4py_backend import (  # noqa (path appending required first)
    GT4PyStencilBackend,  # noqa (path appending required first)
)  # noqa (path appending required first)
from gt4py_benchmarks.scripts.benchmark import (  # noqa (path appending required first)
    run_benchmark,  # noqa (path appending required first)
)  # noqa (path appending required first)


def has_cupy():
    """Test whether `cupy` is importable."""
    try:
        import cupy  # noqa (import is required to determine wether to run tests for cuda backends)

        return True, None
    except ImportError as err:
        return False, err


class AdvectionDiffusionSuite:
    """An example benchmark that times the performance of advection-diffusion."""

    # ~ params = ([16, 32, 64, 128], ["gtx86", "gtmc", "gtcuda", "numpy"])
    params = ([16], ["gtc:gt:cpu_kfirst"])
    param_names = ("size", "backend")
    timeout = 600

    def setup(self, size, backend):
        """Set up the simulation according to the given parameters."""
        self.simulation = None
        cupy_ok, cupy_import_err = has_cupy()
        if not cupy_ok and backend in ["gtcuda"]:
            print(f"cupy not importable: {cupy_import_err}", file=sys.stderr)
            raise NotImplementedError()
        self.runtime = SingleNodeRuntime(
            stencil_backend=GT4PyStencilBackend(gt4py_backend=backend, dtype="float")
        )
        self.runs = 10

    def time_run(self, size, backend):
        """Run the benchmark simulation loop."""
        run_benchmark(self.runtime, [size, size, size], self.runs)
