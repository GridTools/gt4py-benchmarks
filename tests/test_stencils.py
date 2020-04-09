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
        # "numpy",
        "gtx86",
        # "gtmc",
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


class DiffusionSim:
    def __init__(self, shape, *, stencil, reference):
        self.dtype = TEST_DTYPE
        self.domain = analytical.DOMAIN
        self.shape = shape
        self.dspace = numpy.array(self.domain, dtype=float) / numpy.array(self.shape, dtype=float)
        self.max_time = 1e-2  # 1e-3 + 2e-5
        self.time_step = 1e-3

        self.stencil = stencil
        self.reference = reference

        storage_b = (
            self.stencil.storage_builder()
            .backend(self.stencil.backend)
            .dtype(self.stencil.SCALAR_T)
            .default_origin(TEST_ORIGIN)
        )

        self.data = storage_b.from_array(numpy.fromfunction(self.get_reference, shape=self.shape))
        self.data1 = storage_b.from_array(numpy.fromfunction(self.get_reference, shape=self.shape))

    def map_to_domain(self, i, j, k):
        return analytical.map_domain(i, j, k, resolution=self.shape, domain=self.domain)

    def get_reference(self, i, j, k, time=0):
        return self.reference(
            *self.map_to_domain(i, j, k), diffusion_coeff=self.stencil.coeff, time=time
        )

    def swap_data(self):
        tmp = self.data
        self.data = self.data1
        self.data1 = tmp

    def run(self):
        time = 0
        while time < self.max_time:
            self.stencil(
                self.data1, self.data, dt=self.time_step,
            )
            self.swap_data()
            time += self.time_step

    @classmethod
    def from_direction(cls, direction, *, shape, backend, coeff=0.05):
        stencil_cls, reference = None, None
        if direction == "horizontal":
            stencil_cls, reference = diffusion.Horizontal, analytical.horizontal_diffusion
        elif direction == "vertical":
            stencil_cls, reference = diffusion.Vertical, analytical.vertical_diffusion

        dtype = stencil_cls.SCALAR_T
        dspace = numpy.array(analytical.DOMAIN, dtype=type) / numpy.array(shape, dtype=type)
        stencil_args = {"backend": backend, "dspace": dspace, "coeff": coeff}

        return cls(shape, stencil=stencil_cls(**stencil_args), reference=reference)

    def __repr__(self):
        return f"<DiffusionSim: stencil = {self.stencil.name()}, ref = {self.reference.__name__}>"


@pytest.fixture(params=["horizontal", "vertical"])
def diffusion_dir(request):
    yield request.param


@pytest.fixture
def diffusion_sim(test_backend, diffusion_dir):
    shape = (16, 16, 16)
    sim = DiffusionSim.from_direction(diffusion_dir, shape=shape, backend=test_backend, coeff=0.05)
    yield sim


def test_diff(test_backend, diffusion_sim):
    """Test horizontal diffusion stencil stays within tolerance of reference."""
    import copy

    timestep = 1e-3
    sim = diffusion_sim
    start_data = copy.deepcopy(sim.data)
    expected = numpy.fromfunction(
        functools.partial(sim.get_reference, time=sim.max_time), shape=sim.shape
    )
    sim.run()
    errors = numpy.abs(expected[3:-3, 3:-3, 1:-1] - sim.data[3:-3, 3:-3, 1:-1])
    mean_change = numpy.abs(start_data - expected).mean()
    print(f"The mean_abs(exact[t] - exact[t=0]) is: {mean_change}")
    print(f"The mean error is: {errors.mean()}")
    assert (sim.data != start_data).any() and (sim.data1 != start_data).any()
    assert errors.max() < 1e-5
