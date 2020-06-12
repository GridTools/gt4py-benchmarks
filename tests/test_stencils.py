"""Test the stencils for correctness agains the exact validation functions."""
import copy
import functools

from gt4py import gtscript
import numpy
import pytest

from gt4py_benchmarks.stencils import diffusion, advection
from gt4py_benchmarks.verification import analytical


TEST_BACKEND = "debug"
TEST_VERBOSE = True
TEST_STENCIL = functools.update_wrapper(
    functools.partial(gtscript.stencil, backend=TEST_BACKEND), gtscript.stencil
)
TEST_ORIGIN = (3, 3, 0)
TEST_DTYPE = numpy.float64


def has_cupy():
    """Test whether `cupy` is importable."""
    try:
        import cupy  # noqa (import is required to determine wether to run tests for cuda backends)

        return True
    except ImportError:
        return False


@pytest.fixture(
    params=[
        "debug",
        "numpy",
        "gtx86",
        "gtmc",
        pytest.param(
            "gtcuda",
            marks=pytest.mark.skipif(
                not has_cupy(), reason="cupy dependency for gtcuda backend not installed."
            ),
        ),
    ]
)
def test_backend(request):
    """Parametrize by backend name."""
    yield request.param


class Simulation:
    """Wrap stencil and reference into a simulation to verify convergence."""

    def __init__(self, test_spec: dict, *, backend: str):
        """Construct from a test specification and the backend fixture."""
        self.domain = analytical.DOMAIN
        self.time_step = 1e-3
        self.max_time = 1e-2
        self.shape = (16, 16, 16)
        self.backend_name = backend
        self.tolerance = test_spec["tolerance"]
        dspace = numpy.array(analytical.DOMAIN, dtype=numpy.float64) / numpy.array(
            self.shape, dtype=numpy.float64
        )
        stencil_args = {
            "backend": self.backend_name,
            "dspace": dspace,
            "time_step": self.time_step,
        }
        stencil_args.update(test_spec.get("extra-args", {}))
        self.extra_args = test_spec.get("extra-args", {})
        self.stencil = test_spec["stencil"](**stencil_args)
        self.reference = test_spec["reference"]
        storage_b = self.stencil.storage_builder().default_origin(TEST_ORIGIN)

        self.data = storage_b.from_array(numpy.fromfunction(self.get_reference, shape=self.shape))
        self.data1 = copy.deepcopy(self.data)
        self._initial_state = copy.deepcopy(self.data)
        self._expected = numpy.fromfunction(
            functools.partial(self.get_reference, time=self.max_time), shape=self.shape
        )

    def run(self):
        """Run the simulation until `self.max_time`."""
        time = 0
        while time <= self.max_time:
            self.stencil(
                self.data1, self.data, dt=self.time_step,
            )
            self._swap_data()
            time += self.time_step

    def __repr__(self):
        """Build a helpful string representation in case a test fails."""
        return (
            f"<Simulation: stencil = {self.stencil.name()} "
            f"@ {self.backend_name} vs. {self.reference.__name__}>"
        )

    def map_to_domain(self, i: int, j: int, k: int):
        """Map from IJK coordinates to XYZ."""
        return analytical.map_domain(i, j, k, resolution=self.shape, domain=self.domain)

    def get_reference(self, i: int, j: int, k: int, time: float = 0.0):
        """Get reference values at IJK grid points."""
        return self.reference(*self.map_to_domain(i, j, k), time=time, **self.extra_args)

    def _swap_data(self):
        """Swap input and output buffers after time step."""
        tmp = self.data
        self.data = self.data1
        self.data1 = tmp

    @property
    def expected(self):
        """Construct the reference values on the grid at `t=max_time`."""
        return self._expected[3:-3, 3:-3, 1:-1]

    @property
    def result(self):
        """Return the current result at `t=current_time`."""
        return self.data[3:-3, 3:-3, 1:-1]

    @property
    def initial(self):
        """Return the initial state."""
        return self._initial_state[3:-3, 3:-3, 1:-1]

    @property
    def change(self):
        """Return the absolute differences between initial and current state."""
        return numpy.abs(self.initial - self.result)

    @property
    def expected_change(self):
        """Return the absolute difference between the expected result and the initial state."""
        return numpy.abs(self.expected - self.initial)

    @property
    def errors(self):
        """Return the absolute differences between current and expected state."""
        return numpy.abs(self.expected - self.result)


CASES = {
    "horizontal-diffusion": {
        "stencil": diffusion.Horizontal,
        "reference": analytical.horizontal_diffusion,
        "tolerance": 1e-5,
        "extra-args": {"coeff": 0.05},
    },
    "vertical-diffusion": {
        "stencil": diffusion.Vertical,
        "reference": analytical.vertical_diffusion,
        "tolerance": 5e-5,
        "extra-args": {"coeff": 0.05},
    },
    "full-diffusion": {
        "stencil": diffusion.Full,
        "reference": analytical.full_diffusion,
        "tolerance": 2e-3,
        "extra-args": {"coeff": 0.05},
    },
    "horizontal-advection": {
        "stencil": advection.Horizontal,
        "reference": analytical.horizontal_advection,
        "tolerance": 2e-3,
    },
    "vertical-advection": {
        "stencil": advection.Vertical,
        "reference": analytical.vertical_advection,
        "tolerance": 3e-3,
    },
}


@pytest.fixture(params=CASES.items())
def simulation_spec(request):
    """Parametrize by test case simulation specification."""
    yield request.param


@pytest.fixture
def simulation(test_backend, simulation_spec):
    """Yield all the stencil simulations to be tested with associated accuracy tolerance."""
    yield Simulation(simulation_spec[1], backend=test_backend)


def test_stencil(simulation):
    """Test that diffusion stencil stays within tolerance of reference."""
    sim = simulation
    sim.run()
    mean_error = sim.errors.mean()
    mean_change = sim.expected_change.mean()
    max_data_change = sim.change.max()
    print(f"The mean_abs(exact[t] - exact[t=0]) is: {mean_change}")
    print(f"The max_abs(approx[t] - exact[t=0]) is: {max_data_change}")
    print(f"The mean error is: {mean_error}")
    assert max_data_change > 1e-28
    assert sim.errors.max() == pytest.approx(0.0, abs=sim.tolerance)
