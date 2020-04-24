from numpy import float64, array
from functools import update_wrapper, partial

from gt4py.gtscript import Field, computation, interval, PARALLEL, stencil, function

from gt4py_benchmarks.config import GT_BACKEND, STENCIL_VERBOSE
from gt4py_benchmarks.stencils.tooling import AbstractSubstencil, using


DTYPE = float64
_F64 = Field[DTYPE]


class Forward(AbstractSubstencil):
    """
    Tridiagonal Forward sub-stencil.

    Usage in Stencil::

        with computation(FORWARD):
            interval(0, 1):
                c, d = forward_0_1(a, b, c, d)
            interval(1, None):
                c, d = forward_1_last(a, b, c, d)

    Args:
        * a, b: input fields
        * c, d: in/out fields
    """

    @classmethod
    def name(cls):
        return "forward"

    @staticmethod
    def forward_0_1(a, b, c, d):
        c = c / b
        d = d / b
        return c, d

    @staticmethod
    def forward_1_last(a, b, c, d):
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
        return c, d


# Forward.register_globally(globals())
# print(globals()['forward_0_1'])


@using(globals(), Forward)
class PeriodicForward1(AbstractSubstencil):
    """
    Tridiagonal Periodic Forward 1 sub-stencil.

    Usage in Stencil::

        with computation(FORWARD):
            interval(0, 1):
                b, c, d = periodic_forward1_0_1(a, b, c, d, alpha, beta, gamma)
            interval(1, -1):
                periodic_forward1_1_m1(a, b, c, d, alpha, beta, gamma)
            interval(-1, None):
                periodic_forward1_m1_last(a, b, c, d, alpha, beta, gamma)

    Args:
        * a: input field
        * alpha, beta, gamma: input fields (storage_ij_t)
        * b, c, d: in/out fields

    GTBench Docs::

        /*
        * tridiagonal_periodic1:
        * b[0] = b[0] - gamma
        * b[-1] = b[-1] - alpha * beta / gamma
        * x = tridiagonal_solve(a, b, c, d)
        */
    """

    @classmethod
    def name(cls):
        return "periodic_forward1"

    @classmethod
    def uses(cls):
        return [Forward]

    @staticmethod
    def periodic_forward1_0_1(a, b, c, d, alpha, beta, gamma):
        b = b - gamma

        ## inlining
        ## c, d = forward_0_1(a, b, c, d)
        c = c / b
        d = d / b
        ## end inlining

        return b, c, d

    @staticmethod
    def periodic_forward1_1_m1(a, b, c, d, alpha, beta, gamma):
        ## inlining
        ## c, d = forward_1_last(a, b, c, d)
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
        ## end inlining
        return b, c, d

    @staticmethod
    def periodic_forward1_m1_last(a, b, c, d, alpha, beta, gamma):
        b = b - alpha * beta / gamma

        ## inlining
        ## c, d = forward_1_last(a, b, c, d)
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
        ## end inlining

        return b, c, d


@using(globals(), Forward)
class PeriodicForward2(AbstractSubstencil):
    """
    Tridiagonal Periodic Forward 2 sub-stencil.

    Usage in Stencil::

        with computation(FORWARD):
            interval(0, 1):
                b, c, d = periodic_forward2_0_1(a, b, c, d, alpha, beta, gamma)
            interval(1, -1):
                periodic_forward2_1_m1(a, b, c, d, alpha, beta, gamma)
            interval(-1, None):
                periodic_forward2_m1_last(a, b, c, d, alpha, beta, gamma)

    Args:
        * a, b: input fields
        * alpha, gamma: input_fields (storage_ij_t)
        * c, u: in/out fields

    GTBench Docs::

        /*
        * tridiagonal_periodic2:
        * u = np.zeros_like(a)
        * u[0] = gamma
        * u[-1] = alpha
        * z = tridiagonal_solve(a, b, c, u)
        * fact = (x[0] + beta * x[-1] / gamma) / (1 + z[0] + beta * z[-1] / gamma)
        */
    """

    @classmethod
    def name(cls):
        return "periodic_forward2"

    @classmethod
    def uses(cls):
        return [Forward]

    @staticmethod
    def periodic_forward2_0_1(a, b, c, u, alpha, gamma):
        u = gamma
        c, u = forward_0_1(a, b, c, u)
        return c, u

    @staticmethod
    def periodic_forward2_1_m1(a, b, c, u, alpha, gamma):
        u = 0

        ## inlining
        ## c, u = forward_1_last(a, b, c, u)
        # c = c / (b - c[0, 0, -1] * a)
        # u = (u - a * u[0, 0, -1]) / (b - c[0, 0, -1] * a)
        ## end inlining forward_1_last

        return u

    @staticmethod
    def periodic_forward2_m1_last(a, b, c, u, alpha, gamma):
        u = alpha

        # inlining
        ## c, u = forward_1_last(a, b, c, u)
        # c = c / (b - c[0, 0, -1] * a)
        # u = (u - a * u[0, 0, -1]) / (b - c[0, 0, -1] * a)
        ## end inlining forward_1_last

        return u


class Backward(AbstractSubstencil):
    """
    Tridiagonal backward sub-stencil.

    Usage in Stencil::

        with computation(BACKWARD):
            with interval(1, None):
                out = backward_m1_last(d)
            with interval(0, 1):
                out = backward_0_m1(out, c, d)

    the last layer is initialized in `backward_m1_last` and depended on
    in `backward_0_m1`.

    Args:
        * c, d: input fields
        * out: pure out field with vertical backward dependency.
    """

    @classmethod
    def name(cls):
        return "backward"

    @staticmethod
    def backward_0_m1(out, c, d):
        return d - c * out[0, 0, 1]

    @staticmethod
    def backward_m1_last(d):
        return d


@using(globals(), Backward)
class PeriodicBackward1(AbstractSubstencil):
    """Backward with a different name."""

    @classmethod
    def name(cls):
        return "periodic_backward1"

    @classmethod
    def uses(cls):
        return [Backward]

    @staticmethod
    def periodic_backward1_0_m1(out, c, d):
        return backward_0_m1(out, c, d)

    @staticmethod
    def periodic_backward1_m1_last(d):
        return backward_m1_last(d)


@using(globals(), Backward)
class PeriodicBackward2(AbstractSubstencil):
    """
    Periodic Backward 2 sub-stencil.

    Usage in Stencil::

        with computation(BACKWARD):
            interval(-1, None):
                z, z_top, x_top = periodic_backward2_m1_last(d)
            interval(1, -1):
                z = periodic_backward2_1_m1(z, c, d)
            interval(0, 1):
                z, fact = periodic_backward2_0_1(z, c, d, x, beta, gamma, fact, z_top, x_top)

    Args:
        * x, beta, gamma: input fields
        * c, d: in/out fields
        * z: output field with backward dependency
        * fact, z_top, x_top: pure output fields
    """

    @classmethod
    def name(cls):
        return "periodic_backward2"

    @classmethod
    def uses(cls):
        return [Backward]

    @staticmethod
    def periodic_backward2_0_1(*, z, c, d, x, beta, gamma, z_top, x_top):
        fact = (x + beta * x_top / gamma) / (1.0 + z + beta * z_top / gamma)
        return fact

    @staticmethod
    def periodic_backward2_1_m1(*, z, c, d):
        z = backward_0_m1(z, c, d)
        return z

    @staticmethod
    def periodic_backward2_m1_last(*, d, x):
        z = backward_m1_last(d)
        z_top = z
        x_top = x
        return z, z_top, x_top


class Periodic3(AbstractSubstencil):
    """
    Periodic Parallel 3 sub-stencil.

    Args:
        * x, z, fact: input fields
        * data_out (only in original) pure output field
    """

    @classmethod
    def name(cls):
        return "periodic3"

    @staticmethod
    def periodic3_full(x, z, fact):
        return x - fact * z
