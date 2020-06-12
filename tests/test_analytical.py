"""Test analytical solvers agains the output from GTBench."""
import pathlib
import functools
import typing

import numpy
import pytest

from gt4py_benchmarks.verification import analytical


THIS_DIR = pathlib.Path(__file__).parent
DATA_DIR = THIS_DIR / ".." / "data"

CASES = [
    ("h_diff", analytical.horizontal_diffusion),
    ("v_diff", analytical.vertical_diffusion),
    ("f_diff", analytical.full_diffusion),
    ("h_adv", analytical.horizontal_advection),
    ("v_adv", analytical.vertical_advection),
    ("f_adv", analytical.full_advection),
    ("adv_diff", analytical.advection_diffusion),
]


@pytest.fixture
def time():
    """Parametrize on the simulation time the snapshot was taken from GTBench."""
    yield 0.042


@pytest.fixture
def diffusion_coeff():
    """Parametrize on the diffusion coefficient used in the snapshot from GTBench."""
    yield 0.05


@pytest.fixture(params=CASES)
def data_solver(request, diffusion_coeff):
    """Parametrize on all available analytical solvers."""
    name, solver = request.param
    filename = f"{name}_t0_042.npy"
    coeff = None
    if "diff" in name:
        coeff = diffusion_coeff
    yield filename, solver, coeff


def domain_mapped(
    func: typing.Callable[..., float],
    map_domain: typing.Callable[[int, int, int], typing.Tuple[float]],
) -> typing.Callable[..., float]:
    """Map continuously defined function onto discreete grid."""

    def wrapped(i: int, j: int, k: int, *args, **kwargs):
        return func(*map_domain(i, j, k), *args, **kwargs)

    return wrapped


def test_solver(data_solver, time):
    """Check the analytical verification function agains a snapshot of data from GTBench."""
    ref_file, solver, coeff = data_solver
    gtbench_ref = numpy.load(DATA_DIR / ref_file)
    resolution = gtbench_ref.shape
    print(f"Resolution is: {resolution}")
    domain = analytical.DOMAIN
    if solver == analytical.advection_diffusion:
        domain = analytical.AD_DOMAIN
    map_domain = functools.partial(analytical.map_domain, resolution=resolution, domain=domain)
    kwargs = {}
    if coeff:
        kwargs["coeff"] = coeff
    get_value = functools.partial(domain_mapped(solver, map_domain), time=time, **kwargs)
    data = numpy.fromfunction(get_value, shape=resolution)
    errs = abs(gtbench_ref - data)
    max_err = errs.max()
    assert max_err < 1e-6
