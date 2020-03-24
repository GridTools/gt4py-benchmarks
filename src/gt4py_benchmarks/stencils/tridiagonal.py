from numpy import float64, array
from functools import update_wrapper, partial

from gt4py.gtscript import Field, computation, interval, PARALLEL, stencil, function

from gt4py_benchmarks.config import GT_BACKEND, STENCIL_VERBOSE


DTYPE = float64
_F64 = Field[DTYPE]
STENCIL = update_wrapper(partial(stencil, backend=GT_BACKEND, verbose=STENCIL_VERBOSE))


@function
def forward_first(a, b, c, d):
    c_out = c / (b - c[0, 0, -1] * a)
    d_out = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
    return c_out, d_out


@function
def forward_1_0(a, b, c, d):
    c_out = c / b
    d_out = d / b
    return c_out, d_out


@function
def backward_0_m1(out, c, d):
    return d - c * out[0, 0, 1]


@function
def backward_last(out, c, d):
    return d


@function
def periodic_forward1_first(a, b, c, d, alpha, beta, gamma):
    b_out = b - gamma
    c_out, d_out = forward_first(a, b, c, d)
    return b_out, c_out, d_out


@function
def periodic_forward1_1_m1(a, b, c, d, alpha, beta, gamma):
    return b


@function
def periodic_forward1_last(a, b, c, d, alpha, beta, gamma):
    return b - alpha * beta / gamma


periodic_backward1_0_m1 = backward_0_m1


periodic_backward1_last = backward_last


@function
def periodic_forward2_first(a, b, c, u, alpha, gamma):
    u_out = gamma
    c_out, u_out = forward_first(a, b, c, u_out)
    return c_out, u_out


@function
def periodic_forward2_1_m1(a, b, c, u, alpha, gamma):
    return DTYPE(0)


@function
def periodic_forward2_last(a, b, c, u, alpha, gamma):
    return alpha


@function
def periodic_backward2_first(z, c, d, x, beta, gamma, fact, z_top, x_top):
    z_out = backward_0_m1(z, c, d)
    fact_out = x + beta * x_top / gamma / (DTYPE(1) + z_out + beta * z_top / gamma)
    return z_out, fact_out, z_top, x_top


@function
def periodic_backward2_1_m1(z, c, d, x, beta, gamma, fact, z_top, x_top):
    z_out = backward_0_m1(z, c, d)
    return z_out, fact, z_top, x_top


@function
def periodic_backward2_last(z, c, d, x, beta, gamma, fact, z_top, x_top):
    z_out = backward_last(z, c, d)
    z_top_out = z_out
    x_top_out = x
    return z_out, fact, z_top, x_top


@function
def periodic3_full(x, z, fact):
    return x - fact * z
