"""Test the stencils for correctness agains the exact validation functions."""
import functools

from gt4py import gtscript
import numpy
import pytest

from gt4py_benchmarks.stencils import diffusion, advection
from gt4py_benchmarks.verification import analytical
from gt4py_benchmarks.simulations import Simulation, RkAdvSimulation, AdvDiffSimulation


TEST_BACKEND = "debug"
TEST_VERBOSE = True
TEST_STENCIL = functools.update_wrapper(
    functools.partial(gtscript.stencil, backend=TEST_BACKEND), gtscript.stencil
)
TEST_DTYPE = numpy.float64


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
    "full-advection": {
        "stencil": advection.Full,
        "reference": analytical.full_advection,
        "tolerance": 5e-2,
        "subclass": "RkAdvSimulation",
    },
    "advection-diffusion": {
        "stencil": None,
        "reference": analytical.advection_diffusion,
        "tolerance": 1.5e-2,
        "subclass": "AdvDiffSimulation",
        "domain": analytical.AD_DOMAIN,
        "extra-args": {"coeff": 0.05},
    },
}


SimulationSubclassMap = {
    "Simulation": Simulation,
    "RkAdvSimulation": RkAdvSimulation,
    "AdvDiffSimulation": AdvDiffSimulation,
}


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


@pytest.fixture(params=CASES.items(), ids=CASES.keys())
def simulation_spec(request):
    """Parametrize by test case simulation specification."""
    yield request.param


@pytest.fixture
def simulation(test_backend, simulation_spec):
    """Yield all the stencil simulations to be tested with associated accuracy tolerance."""
    simulation_class = SimulationSubclassMap[simulation_spec[1].get("subclass", "Simulation")]
    yield simulation_class(simulation_spec[1], backend=test_backend)


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
    print(f"The max relative error is: {sim.rel_errors.max()}")
    assert max_data_change > 1e-28
    assert sim.errors.max() == pytest.approx(0.0, abs=sim.tolerance)
