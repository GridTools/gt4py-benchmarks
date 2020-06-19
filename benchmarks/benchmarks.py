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

        return True
    except ImportError:
        return False


class AdvectionDiffusionSuite:
    """An example benchmark that times the performance of advection-diffusion."""

    params = ([8, 16, 32, 64], ["gtx86", "gtmc", "gtcuda", "numpy"])
    param_names = ("size", "backend")
    timeout = 600

    def setup(self, size, backend):
        """Set up the simulation according to the given parameters."""
        if not has_cupy() and backend in ["gtcuda"]:
            raise NotImplementedError()
        elif size > 10:
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
        self.simulation.run()

    def time_run(self, size, backend):
        """Run the benchmark simulation loop."""
        self.simulation.run()

    def teardown(self, size, backend):
        """Reset the simulation data and timer."""
        self.simulation.reset()
