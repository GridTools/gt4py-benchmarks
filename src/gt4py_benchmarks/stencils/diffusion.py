from numpy import float64, array
from functools import update_wrapper, partial

from gt4py.gtscript import (
    Field,
    computation,
    interval,
    FORWARD,
    BACKWARD,
    PARALLEL,
    stencil,
    function,
)

from gt4py_benchmarks.config import GT_BACKEND, STENCIL_VERBOSE
from gt4py_benchmarks.stencils import tridiagonal


DTYPE = float64
_F64 = Field[DTYPE]
STENCIL = update_wrapper(partial(stencil, backend=GT_BACKEND, verbose=STENCIL_VERBOSE))


@STENCIL()
def horizontal(out: _F64, inp: _F64, dx: _F64, dy: _F64, dt: _F64, coeff: _F64):
    with computation(PARALLEL), interval(...):
        weights = array([-1 / 90, 5 / 36, -49 / 36, 49 / 36, -5 / 36, 1 / 90], dtype=DTYPE)
        flx_x0 = (
            weights[0] * inp[-3, 0]
            + weights[1] * inp[-2, 0]
            + weights[2] * inp[-1, 0]
            + weights[3] * inp[0, 0]
            + weights[4] * inp[1, 0]
            + weights[5] * inp[2, 0]
        ) / dx
        flx_x1 = (
            weights[0] * inp[-2, 0]
            + weights[1] * inp[-1, 0]
            + weights[2] * inp[0, 0]
            + weights[3] * inp[1, 0]
            + weights[4] * inp[2, 0]
            + weights[5] * inp[3, 0]
        ) / dx
        flx_y0 = (
            weights[0] * inp[0, -3]
            + weights[1] * inp[0, -2]
            + weights[2] * inp[0, -1]
            + weights[3] * inp[0, 0]
            + weights[4] * inp[0, 1]
            + weights[5] * inp[0, 2]
        ) / dy
        flx_y1 = (
            weights[0] * inp[0, -2]
            + weights[1] * inp[0, -1]
            + weights[2] * inp[0, 0]
            + weights[3] * inp[0, 1]
            + weights[4] * inp[0, 2]
            + weights[5] * inp[0, 3]
        ) / dy

        flx_x0 = flx_x0 * (DTYPE(0) if inp - inp[-1, 0] < DTYPE(0) else flx_x0)
        flx_x1 = flx_x1 * (DTYPE(0) if inp[1, 0] - inp < DTYPE(0) else flx_x1)
        flx_y0 = flx_y0 * (DTYPE(0) if inp - inp[0, -1] < DTYPE(0) else flx_y0)
        flx_y1 = flx_y1 * (DTYPE(0) if inp[0, 1] - inp < DTYPE(0) else flx_y1)

        out = inp + coeff * dt * ((flx_x1 - flx_x0) / dx + (flx_y1 - flx_y0) / dy)


@function
def diffusion_w0_last(out: _F64, inp: _F64):
    with computation(PARALLEL), interval(-1, None):
        out = inp


@function
def diffusion_w_forward1_first(
    alpha: _F64,
    beta: _F64,
    gamma: _F64,
    a: _F64,
    b: _F64,
    c: _F64,
    d: _F64,
    data: _F64,
    data_tmp: _F64,
    dz: _F64,
    dt: _F64,
    coeff: _F64,
):
    c_ = -coeff / (DTYPE(2) * dz * dz)
    a_ = c
    b_ = DTYPE(1) / dt - a_ - c_
    d_ = DTYPE(1) / dt * data + DTYPE(0.5) * coeff * (
        data_tmp - DTYPE(2) * data + data[0, 0, 1]
    ) / (dz * dz)
    beta_ = -coeff / (DTYPE(2) * dz * dz)
    alpha_ = beta_
    gamma_ = -b_

    b_, c_, d_ = tridiagonal.periodic_forward1_first(a_, b_, c_, d_, alpha_, beta_, gamma_)

    data_tmp_ = data

    return alpha_, beta_, gamma_, a_, b_, c_, d_, data_tmp_


@function
def diffusion_w_forward1_1_m1(
    alpha: _F64,
    beta: _F64,
    gamma: _F64,
    a: _F64,
    b: _F64,
    c: _F64,
    d: _F64,
    data: _F64,
    data_tmp: _F64,
    dz: _F64,
    dt: _F64,
    coeff: _F64,
):
    c_ = -coeff / (DTYPE(2) * dz * dz)
    a_ = c_
    b_ = DTYPE(1) / dt - a_ - c_
    d_ = DTYPE(1) / dt * data + DTYPE(0.5) * coeff * (
        data[0, 0, -1] - DTYPE(2) * data + data[0, 0, 1]
    ) / (dz * dz)

    return alpha, beta, gamma, a_, b_, c_, d_, data_tmp


@function
def diffusion_w_forward1_last(
    alpha: _F64,
    beta: _F64,
    gamma: _F64,
    a: _F64,
    b: _F64,
    c: _F64,
    d: _F64,
    data: _F64,
    data_tmp: _F64,
    dz: _F64,
    dt: _F64,
    coeff: _F64,
):
    c_ = -coeff / (DTYPE(2) * dz * dz)
    a_ = c_
    b_ = DTYPE(1) / dt - a_ - c_
    d_ = DTYPE(1) / dt * data + DTYPE(0.5) * coeff * (
        data[0, 0, -1] - DTYPE(2) * data + data_tmp
    ) / (dz * dz)

    return alpha, beta, gamma, a_, b_, c_, d_, data_tmp


@STENCIL()
def vertical(data_out, data_in):
    with computation(FORWARD), interval(-1, None):
        data_out = diffusion_w0_last(data_out, data_in)

    with computation(FORWARD):
        with interval(0, 1):
            alpha, beta, gamma, a, b, c, d, data_in_tmp = diffusion_w_forward1_first(
                alpha, beta, gamma, a, b, c, d, data_in, data_in_tmp, dz, dt, coeff
            )
    with computation(FORWARD):
        with interval(1, -1):
            alpha, beta, gamma, a, b, c, d, data_in_tmp = diffusion_w_forward1_1_m1(
                alpha, beta, gamma, a, b, c, d, data_in, data_in_tmp, dz, dt, coeff
            )
            b = tridiagonal.periodic_forward1_1_m1(a, b, c, d, alpha, beta, gamma)
        with interval(1, None):
            c, d = tridiagonal.forward_1_0(a, b, c, d)
    with computation(FORWARD):
        with interval(-1, None):
            alpha, beta, gamma, a, b, c, d, data_in_tmp = diffusion_w_forward1_last(
                alpha, beta, gamma, a, b, c, d, data_in, data_in_tmp, dz, dt, coeff
            )
            b = tridiagonal.periodic_forward1_last(a, b, c, d, alpha, beta, gamma)
        with interval(1, None):
            c, d = tridiagonal.forward_1_0(a, b, c, d)

    with computation(BACKWARD):
        with interval(0, -1):
            x = tridiagonal.periodic_backward1_0_m1(x, c, d)
        with interval(-1, None):
            x = tridiagonal.periodic_backward1_last(x, c, d)

    with computation(FORWARD):
        with interval(0, 1):
            c, d = tridiagonal.periodic_forward2_first(a, b, c, d, alpha, gamma)
    with computation(FORWARD):
        with interval(1, -1):
            d = tridiagonal.periodic_forward2_1_m1(a, b, c, d, alpha, gamma)
        with interval(1, None):
            c, d = tridiagonal.forward_1_0(a, b, c, d)
    with computation(FORWARD):
        with interval(-1, None):
            d = tridiagonal.periodic_forward2_last(a, b, c, d, alpha, gamma)
        with interval(1, None):
            c, d = tridiagonal.forward_1_0(a, b, c, d)

    with computation(BACKWARD):
        with interval(0, 1):
            z, fact, z_top, x_top = tridiagonal.periodic_backward2_first(
                z, c, d, x, beta, gamma, fact, z_top, x_top
            )
        with interval(1, -1):
            z, fact, z_top, x_top = tridiagonal.periodic_backward2_1_m1(
                z, c, d, x, beta, gamma, fact, z_top, x_top
            )
        with interval(-1, None):
            z, fact, z_top, x_top = tridiagonal.periodic_backward2_last(
                z, c, d, x, beta, gamma, fact, z_top, x_top
            )

    with computation(PARALLEL), interval(...):
        data_out = tridiagonal.periodic3_full(data_out, x, z, fact, data_in, dt)
