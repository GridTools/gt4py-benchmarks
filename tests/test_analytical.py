import pathlib
import functools

import numpy
import pytest

from gt4py_benchmarks.verification import analytical


THIS_DIR = pathlib.Path(__file__).parent
DATA_DIR = THIS_DIR / ".." / "data"


@pytest.fixture
def gtbench_hdiff():
    yield numpy.load(DATA_DIR / "h_diff_t0_042.npy")


@pytest.fixture
def time():
    yield 0.042


@pytest.fixture
def diffusion_coeff():
    yield 0.05


def test_hdiff(gtbench_hdiff, time, diffusion_coeff):
    resolution = gtbench_hdiff.shape
    print(f"Resolution is: {resolution}")
    map_domain = functools.partial(
        analytical.map_domain, resolution=resolution, domain=analytical.DOMAIN
    )
    get_value = lambda i, j, k: analytical.horizontal_diffusion(
        *map_domain(i, j, k), diffusion_coeff=diffusion_coeff, time=time
    )
    data = numpy.fromfunction(get_value, shape=resolution)
    errs = gtbench_hdiff - data
    max_err = errs.max()
    assert max_err < 1e-6
