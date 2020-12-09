import itertools
import functools

import numpy as np
import pytest

from gt4py_benchmarks.verification import analytical, convergence
from gt4py_benchmarks.runtime.runtimes import SingleNodeRuntime
from gt4py_benchmarks.numerics.stencil_backends import GT4PyStencilBackend, GTBenchStencilBackend


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            functools.partial(GT4PyStencilBackend, gt4py_backend="debug"), marks=pytest.mark.slow
        ),
        pytest.param(
            functools.partial(GT4PyStencilBackend, gt4py_backend="numpy"), marks=pytest.mark.slow
        ),
        functools.partial(GT4PyStencilBackend, gt4py_backend="gtx86"),
        functools.partial(GT4PyStencilBackend, gt4py_backend="gtmc"),
        pytest.param(
            functools.partial(GT4PyStencilBackend, gt4py_backend="gtcuda"),
            marks=pytest.mark.requires_gpu,
        ),
        functools.partial(GT4PyStencilBackend, gt4py_backend="dacex86"),
        pytest.param(
            functools.partial(GT4PyStencilBackend, gt4py_backend="dacecuda"),
            marks=pytest.mark.requires_gpu,
        ),
        functools.partial(GTBenchStencilBackend, gtbench_backend="cpu_ifirst"),
        functools.partial(GTBenchStencilBackend, gtbench_backend="cpu_kfirst"),
        pytest.param(
            functools.partial(GTBenchStencilBackend, gtbench_backend="gpu"),
            marks=pytest.mark.requires_gpu,
        ),
    ],
    ids=lambda f: f.func.__name__
    + "("
    + ", ".join(f"{k}={v}" for k, v in f.keywords.items())
    + ")",
)
def stencil_backend(dtype, request):
    return request.param(dtype=dtype)


@pytest.fixture(params=[SingleNodeRuntime], ids=lambda rt: rt.__name__)
def runtime(stencil_backend, request):
    return request.param(stencil_backend=stencil_backend)


def check_orders(result, dtype, spatial=2, temporal=1):
    measured_s = result.spatial.orders[-1]
    measured_t = result.temporal.orders[-1]
    if dtype == "float32":
        atol_s = atol_t = 0.03
        rtol_s = rtol_t = 0.04
    else:
        atol_s = atol_t = 0
        rtol_s = 0.01
        rtol_t = 0.02

    assert np.isclose(
        measured_s, spatial, rtol=rtol_s, atol=atol_s
    ), "spatial order does not match expected value"
    assert np.isclose(
        measured_t, temporal, rtol=rtol_t, atol=atol_t
    ), "temporal order does not match expected value"


def test_hdiff(runtime):
    result = convergence.default_convergence_tests(runtime).hdiff()
    print(result)
    check_orders(result, runtime.stencil_backend.dtype, spatial=6)


def test_vdiff(runtime):
    result = convergence.default_convergence_tests(runtime).vdiff()
    print(result)
    check_orders(result, runtime.stencil_backend.dtype, temporal=2)


def test_diff(runtime):
    result = convergence.default_convergence_tests(runtime).diff()
    print(result)
    check_orders(result, runtime.stencil_backend.dtype)


def test_hadv(runtime):
    result = convergence.default_convergence_tests(runtime).hadv()
    print(result)
    check_orders(result, runtime.stencil_backend.dtype, spatial=5)


def test_vadv(runtime):
    result = convergence.default_convergence_tests(runtime).vadv()
    print(result)
    check_orders(result, runtime.stencil_backend.dtype, temporal=2)


def test_rkadv(runtime):
    result = convergence.default_convergence_tests(runtime).rkadv()
    print(result)
    check_orders(result, runtime.stencil_backend.dtype)


def test_advdiff(runtime):
    result = convergence.default_convergence_tests(runtime).advdiff()
    print(result)
    check_orders(result, runtime.stencil_backend.dtype)
