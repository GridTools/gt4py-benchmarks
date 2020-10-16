import itertools
import functools

import numpy as np
import pytest

from gt4py_benchmarks.verification import analytical, convergence
from gt4py_benchmarks.runtime.runtimes import SingleNodeRuntime
from gt4py_benchmarks.numerics.stencil_backends import GT4PyStencilBackend


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        # functools.partial(GT4PyStencilBackend, gt4py_backend="numpy"),
        functools.partial(GT4PyStencilBackend, gt4py_backend="gtx86"),
        functools.partial(GT4PyStencilBackend, gt4py_backend="gtmc"),
        functools.partial(GT4PyStencilBackend, gt4py_backend="dacex86"),
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
        atol_s = 0.05
        atol_t = 0.07
        rtol_s = 0.05 if spatial <= 2 else 0.25
        rtol_t = 0.0
    else:
        atol_s = atol_t = 0.04
        rtol_s = rtol_t = 0.0

    assert np.isclose(
        measured_s, spatial, rtol=rtol_s, atol=atol_s
    ), "spatial order does not match expected value"
    assert np.isclose(
        measured_t, temporal, rtol=rtol_t, atol=atol_t
    ), "temporal order does not match expected value"


def test_hdiff(runtime):
    diffusion_coeff = 0.05
    result = convergence.default_convergence_test(
        runtime,
        analytical.horizontal_diffusion(diffusion_coeff),
        runtime.stencil_backend.hdiff_stepper(diffusion_coeff),
    )
    check_orders(result, runtime.stencil_backend.dtype, spatial=6)


def test_vdiff(runtime):
    diffusion_coeff = 0.05
    result = convergence.default_convergence_test(
        runtime,
        analytical.vertical_diffusion(diffusion_coeff),
        runtime.stencil_backend.vdiff_stepper(diffusion_coeff),
    )
    check_orders(result, runtime.stencil_backend.dtype)


def test_diff(runtime):
    diffusion_coeff = 0.05
    result = convergence.default_convergence_test(
        runtime,
        analytical.full_diffusion(diffusion_coeff),
        runtime.stencil_backend.diff_stepper(diffusion_coeff),
    )
    check_orders(result, runtime.stencil_backend.dtype)


def test_hadv(runtime):
    result = convergence.default_convergence_test(
        runtime, analytical.horizontal_advection(), runtime.stencil_backend.hadv_stepper()
    )
    check_orders(result, runtime.stencil_backend.dtype, spatial=5)


def test_vadv(runtime):
    result = convergence.default_convergence_test(
        runtime, analytical.vertical_advection(), runtime.stencil_backend.vadv_stepper()
    )
    check_orders(result, runtime.stencil_backend.dtype)


def test_rkadv(runtime):
    result = convergence.default_convergence_test(
        runtime, analytical.full_advection(), runtime.stencil_backend.rkadv_stepper()
    )
    check_orders(result, runtime.stencil_backend.dtype)


def test_advdiff(runtime):
    diffusion_coeff = 0.05
    result = convergence.default_convergence_test(
        runtime,
        analytical.advection_diffusion(diffusion_coeff),
        runtime.stencil_backend.advdiff_stepper(diffusion_coeff),
    )
    check_orders(result, runtime.stencil_backend.dtype)
