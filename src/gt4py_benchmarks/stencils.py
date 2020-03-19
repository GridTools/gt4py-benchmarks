from numpy import float64, array
from functools import update_wrapper, partial

from gt4py.gtscript import Field, computation, interval, PARALLEL, stencil

from gt4py_benchmarks.config import GT_BACKEND, STENCIL_VERBOSE


DTYPE = float64
_F64 = Field[DTYPE]
STENCIL = update_wrapper(partial(stencil, backend=GT_BACKEND, verbose=STENCIL_VERBOSE))


@STENCIL()
def stage_horizontal(out: _F64, inp: _F64, dx: _F64, dy: _F64, dt: _F64, coeff: _F64):
    with computation(PARALLEL), interval(...):
        weights = array([-1 / 90, 5 / 36, -49 / 36, 49 / 36, -5 / 36, 1 / 90], dtype=DTYPE)
        flx_x0 = (weights[0] * inp[-3, 0] + weights[1] * inp[-2, 0] + weights[2] * inp[-1, 0] +
                  weights[3] * inp[0, 0] + weights[4] * inp[1, 0] + weights[5] * inp[2, 0]) / dx
        flx_x1 = (weights[0] * inp[-2, 0] + weights[1] * inp[-1, 0] + weights[2] * inp[0, 0] +
                  weights[3] * inp[1, 0] + weights[4] * inp[2, 0] + weights[5] * inp[3, 0]) / dx
        flx_y0 = (weights[0] * inp[0, -3] + weights[1] * inp[0, -2] + weights[2] * inp[0, -1] +
                  weights[3] * inp[0, 0] + weights[4] * inp[0, 1] + weights[5] * inp[0, 2]) / dy
        flx_y1 = (weights[0] * inp[0, -2] + weights[1] * inp[0, -1] + weights[2] * inp[0, 0] +
                  weights[3] * inp[0, 1] + weights[4] * inp[0, 2] + weights[5] * inp[0, 3]) / dy

        flx_x0 *= DTYPE(0) if inp - inp[-1, 0] < DTYPE(0) else flx_x0
        flx_x1 *= DTYPE(0) if inp[1, 0] - inp < DTYPE(0) else flx_x1
        flx_y0 *= DTYPE(0) if inp - inp[0, -1] < DTYPE(0) else flx_y0
        flx_y1 *= DTYPE(0) if inp[0, 1] - inp < DTYPE(0) else flx_y1

        out = inp + coeff * dt * (flx_x1 - flx_x0) / dx + (flx_y1 - flx_y0) / dy)
