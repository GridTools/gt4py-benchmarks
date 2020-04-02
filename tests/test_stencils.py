import functools

from gt4py import gtscript, storage
import numpy
import pytest

from gt4py_benchmarks import config
from gt4py_benchmarks.stencils import diffusion
from gt4py_benchmarks.verification import analytical


TEST_BACKEND = "debug"
TEST_VERBOSE = True
TEST_STENCIL = functools.update_wrapper(
    functools.partial(gtscript.stencil, backend=TEST_BACKEND), gtscript.stencil
)
TEST_ORIGIN = (3, 3, 0)
TEST_DTYPE = numpy.float64


def has_cupy():
    try:
        import cupy

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
    yield request.param


class Simulation:
    def __init__(self, shape, *, backend):
        self.dtype = TEST_DTYPE
        self.domain = analytical.DOMAIN
        self.shape = shape
        self.max_time = 1e-2
        self.time_step = 1e-3
        self.diffusion_coeff = 0.05
        self.backend = backend
        self.stencil = diffusion.Horizontal(backend=self.backend)

        storage_b = (
            self.stencil.storage_builder()
            .backend(self.backend)
            .dtype(self.dtype)
            .default_origin(TEST_ORIGIN)
        )

        self.data = storage_b.from_array(numpy.fromfunction(self.get_reference, shape=self.shape))
        self.data1 = storage_b.from_array(numpy.zeros(self.shape))
        self.delta_x = self.dtype(self.domain[0]) / self.shape[0]
        self.delta_y = self.dtype(self.domain[1]) / self.shape[1]
        self.delta_t = self.time_step

    def map_to_domain(self, i, j, k):
        return analytical.map_domain(i, j, k, resolution=self.shape, domain=self.domain)

    def get_reference(self, i, j, k, time=0):
        return analytical.horizontal_diffusion(
            *self.map_to_domain(i, j, k), diffusion_coeff=self.diffusion_coeff, time=time
        )

    def make_storage(self, array):
        return storage_b.from_array(array)

    def swap_data(self):
        tmp = self.data
        self.data = self.data1
        self.data1 = tmp

    def run(self):
        time = 0
        while time < self.max_time:
            self.stencil(
                self.data1,
                self.data,
                dx=self.delta_x,
                dy=self.delta_y,
                dt=self.time_step,
                coeff=self.diffusion_coeff,
            )
            self.swap_data()
            time += self.time_step


def test_hdiff(test_backend):
    timestep = 1e-3
    shape = (10, 10, 60)
    sim = Simulation(shape, backend=test_backend)
    expected = numpy.fromfunction(
        functools.partial(sim.get_reference, time=sim.max_time), shape=sim.shape
    )
    sim.run()
    max_err = (expected - sim.data).max()
    assert max_err < 1e-2