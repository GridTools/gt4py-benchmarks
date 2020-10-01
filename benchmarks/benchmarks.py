"""GT4Py benchmarking suite."""
import sys
import pathlib

bpath = pathlib.Path(__file__).parent
sys.path.append(str(bpath.parent / "src"))

from gt4py_benchmarks.verification import analytical  # noqa (path appending required first)
from gt4py_benchmarks.simulations import AdvDiffSimulation  # noqa (path appending required first)


def has_cupy():
    """Test whether `cupy` is importable."""
    try:
        import cupy  # noqa (import is required to determine wether to run tests for cuda backends)

        return True, None
    except ImportError as err:
        return False, err


class AdvectionDiffusionSuite:
    """An example benchmark that times the performance of advection-diffusion."""

    params = ([16, 32, 64, 128], ["gtx86", "gtmc", "gtcuda", "numpy"])
    param_names = ("size", "backend")
    timeout = 600

    def setup(self, size, backend):
        """Set up the simulation according to the given parameters."""
        self.simulation = None
        cupy_ok, cupy_import_err = has_cupy()
        if not cupy_ok and backend in ["gtcuda"]:
            print(f"cupy not importable: {cupy_import_err}", file=sys.stderr)
            raise NotImplementedError()
        self.simulation_spec = {
            "stencil": None,
            "reference": analytical.advection_diffusion,
            "tolerance": 1.5e-2,
            "subclass": AdvDiffSimulation,
            "domain": analytical.AD_DOMAIN,
            "extra-args": {"coeff": 0.05},
            "shape": (size, size, 60),
            "max-time": 0.1,
            "time-step": 1e-3,
        }
        self.simulation = AdvDiffSimulation(self.simulation_spec, backend=backend)

    def time_run(self, size, backend):
        """Run the benchmark simulation loop."""
        self.simulation.run()

    def teardown(self, size, backend):
        """Reset the simulation data and timer."""
        if self.simulation is not None:
            self.simulation.reset()
