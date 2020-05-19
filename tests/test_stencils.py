import functools

from gt4py import gtscript, storage
import numpy
import pytest

from gt4py_benchmarks import config
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
    try:
        import cupy

        return True
    except ImportError:
        return False


@pytest.fixture(
    params=[
        "debug",
        "numpy",
        # "gtx86",
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
    def __init__(self, shape, *, stencil, reference, time_step=1e-3, max_time=1e-2):
        self.dtype = TEST_DTYPE
        self.domain = analytical.DOMAIN
        self.shape = shape
        self.dspace = numpy.array(self.domain, dtype=float) / numpy.array(self.shape, dtype=float)
        self.max_time = max_time
        self.time_step = time_step

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
        while time <= self.max_time:
            self.stencil(
                self.data1, self.data, dt=self.time_step,
            )
            self.swap_data()
            time += self.time_step

    @classmethod
    def from_direction(
        cls, direction, *, shape, backend, coeff=0.05, time_step=1e-3, max_time=1e-2
    ):
        stencil_cls, reference = None, None
        if direction == "horizontal":
            stencil_cls, reference = diffusion.Horizontal, analytical.horizontal_diffusion
        elif direction == "vertical":
            stencil_cls, reference = diffusion.Vertical, analytical.vertical_diffusion
        elif direction == "full":
            stencil_cls, reference = diffusion.Full, analytical.full_diffusion

        dtype = stencil_cls.SCALAR_T
        dspace = numpy.array(analytical.DOMAIN, dtype=dtype) / numpy.array(shape, dtype=dtype)
        stencil_args = {
            "backend": backend,
            "dspace": dspace,
            "coeff": coeff,
            "time_step": time_step,
        }

        return cls(
            shape,
            stencil=stencil_cls(**stencil_args),
            reference=reference,
            time_step=time_step,
            max_time=max_time,
        )

    def __repr__(self):
        return f"<DiffusionSim: stencil = {self.stencil.name()}, ref = {self.reference.__name__}>"


class AdvectionSim(DiffusionSim):
    def run(self):
        time = 0
        while time <= self.max_time:
            self.stencil(self.data1, self.data, dt=self.time_step)
            self.swap_data()
            time += self.time_step

    def __repr__(self):
        return f"<AdvectionSim: stencil = {self.stencil.name()}, ref = {self.reference.__name__}>"

    def get_reference(self, i, j, k, time=0):
        return self.reference(*self.map_to_domain(i, j, k), time=time)

    @classmethod
    def from_direction(cls, direction, *, shape, backend, time_step=1e-3, max_time=1e-2):
        stencil_cls, reference = None, None
        if direction == "horizontal":
            stencil_cls, reference = advection.Horizontal, analytical.horizontal_advection
        elif direction == "vertical":
            stencil_cls, reference = advection.Vertical, analytical.vertical_advection

        dtype = stencil_cls.SCALAR_T
        dspace = numpy.array(analytical.DOMAIN, dtype=dtype) / numpy.array(shape, dtype=dtype)
        stencil_args = {"backend": backend, "dspace": dspace, "time_step": time_step}

        return cls(
            shape,
            stencil=stencil_cls(**stencil_args),
            reference=reference,
            time_step=time_step,
            max_time=max_time,
        )


@pytest.fixture(params=[("horizontal", 1e-5), ("vertical", 5e-5), ("full", 2e-3)])
def diffusion_dir_tol(request):
    yield request.param


@pytest.fixture(params=[("horizontal", 2e-3), ("vertical", 3e-3)])
def advection_dir_tol(request):
    yield request.param


@pytest.fixture
def diffusion_sim_tol(test_backend, diffusion_dir_tol):
    shape = (16, 16, 16)
    direction, tolerance = diffusion_dir_tol
    sim = DiffusionSim.from_direction(
        direction, shape=shape, backend=test_backend, coeff=0.05, time_step=1e-3
    )
    yield sim, tolerance


@pytest.fixture
def advection_sim_tol(test_backend, advection_dir_tol):
    shape = (16, 16, 16)
    direction, tolerance = advection_dir_tol
    sim = AdvectionSim.from_direction(
        direction, shape=shape, backend=test_backend, time_step=1e-3, max_time=1e-2
    )
    yield sim, tolerance


@pytest.fixture
def sim_tol(diffusion_sim_tol, advection_sim_tol):
    yield from diffusion_sim_tol
    yield from advection_sim_tol


def test_diff(test_backend, diffusion_sim_tol):
    """Test that diffusion stencil stays within tolerance of reference."""
    import copy

    sim, tolerance = diffusion_sim_tol
    start_data = copy.deepcopy(sim.data)
    expected = numpy.fromfunction(
        functools.partial(sim.get_reference, time=sim.max_time), shape=sim.shape
    )
    sim.run()
    errors = numpy.abs(expected - sim.data)[3:-3, 3:-3, 1:-1]
    mean_change = numpy.abs(start_data - expected)[3:-3, 3:-3, 1:-1].mean()
    print(f"The mean_abs(exact[t] - exact[t=0]) is: {mean_change}")
    print(f"The mean error is: {errors.mean()}")
    assert (sim.data != start_data).any() and (sim.data1 != start_data).any()
    assert errors.max() < tolerance


def test_adv(test_backend, advection_sim_tol):
    """Test that diffusion stencil stays within tolerance of reference."""
    import copy

    sim, tolerance = advection_sim_tol
    start_data = copy.deepcopy(sim.data)
    expected = numpy.fromfunction(
        functools.partial(sim.get_reference, time=sim.max_time), shape=sim.shape
    )
    sim.run()
    errors = numpy.abs(expected - sim.data)[3:-3, 3:-3, 1:-1]
    mean_change = numpy.abs(start_data - expected)[3:-3, 3:-3, 1:-1].mean()
    max_data_change = numpy.abs(start_data - sim.data)[3:-3, 3:-3, 1:-1].max()
    max_data1_change = numpy.abs(start_data - sim.data1)[3:-3, 3:-3, 1:-1].max()
    print(f"The mean_abs(exact[t] - exact[t=0]) is: {mean_change}")
    print(f"The max_abs(approx[t] - exact[t=0]) is: {max_data_change}")
    print(f"The mean error is: {errors.mean()}")
    assert (max_data_change > 1e-28) and (max_data1_change > 1e-28)
    assert errors.max() < tolerance
