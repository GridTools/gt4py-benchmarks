"""Basic building block substencils."""
from gt4py_benchmarks.stencils.tooling import AbstractSubstencil, using


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
        """Declare name."""
        return "forward"

    @staticmethod
    def forward_0_1(a, b, c, d):
        """Calculate the subroutine for the first vertical layer."""
        c = c / b
        d = d / b
        return c, d

    @staticmethod
    def forward_1_last(a, b, c, d):
        """Calculate the subroutine for all but the first vertical layer."""
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
        return c, d


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
        """Declare name."""
        return "periodic_forward1"

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [Forward]

    @staticmethod
    def periodic_forward1_0_1(a, b, c, d, alpha, beta, gamma):
        """Calculate the subroutine for the first vertical layer."""
        b = b - gamma
        c = c / b
        d = d / b
        return b, c, d

    @staticmethod
    def periodic_forward1_1_m1(a, b, c, d, alpha, beta, gamma):
        """Calculate the subroutine for the middle vertical layers."""
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
        return b, c, d

    @staticmethod
    def periodic_forward1_m1_last(a, b, c, d, alpha, beta, gamma):
        """Calculate the subroutine for the last vertical layer."""
        b = b - alpha * beta / gamma
        c = c / (b - c[0, 0, -1] * a)
        d = (d - a * d[0, 0, -1]) / (b - c[0, 0, -1] * a)
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
        """Declare the name."""
        return "periodic_forward2"

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [Forward]

    @staticmethod
    def periodic_forward2_0_1(a, b, c, u, alpha, gamma):
        """Calculate the subroutine for the first vertical layer."""
        u = gamma
        c, u = forward_0_1(a, b, c, u)  # noqa (handled by stencil composition system)
        return c, u

    @staticmethod
    def periodic_forward2_1_m1(a, b, c, u, alpha, gamma):
        """Calculate the subroutine for the middle vertical layers."""
        u = 0
        return u

    @staticmethod
    def periodic_forward2_m1_last(a, b, c, u, alpha, gamma):
        """Calculate the subroutine for the last vertical layer."""
        u = alpha
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
        """Declare name."""
        return "backward"

    @staticmethod
    def backward_0_m1(out, c, d):
        """Calculate the subroutine for all but the last vertical layer."""
        return d - c * out[0, 0, 1]

    @staticmethod
    def backward_m1_last(d):
        """Calculate the subroutine for the last vertical layer."""
        return d


@using(globals(), Backward)
class PeriodicBackward1(AbstractSubstencil):
    """Backward with a different name."""

    @classmethod
    def name(cls):
        """Declare name."""
        return "periodic_backward1"

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [Backward]

    @staticmethod
    def periodic_backward1_0_m1(out, c, d):
        """Calculate the subroutine for all but the last vertical layer."""
        return backward_0_m1(out, c, d)  # noqa (handled by stencil composition system)

    @staticmethod
    def periodic_backward1_m1_last(d):
        """Calculate the subroutine for the last vertical layer."""
        return backward_m1_last(d)  # noqa (handled by stencil composition system)


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
        """Declare name."""
        return "periodic_backward2"

    @classmethod
    def uses(cls):
        """Declare substencil usage."""
        return [Backward]

    @staticmethod
    def periodic_backward2_0_1(*, z, c, d, x, beta, gamma, z_top, x_top):
        """Calculate the subroutine for the first vertical layer."""
        fact = (x + beta * x_top / gamma) / (1.0 + z + beta * z_top / gamma)
        return fact

    @staticmethod
    def periodic_backward2_1_m1(*, z, c, d):
        """Calculate the subroutine for the middle vertical layers."""
        z = backward_0_m1(z, c, d)  # noqa (handled by stencil composition system)
        return z

    @staticmethod
    def periodic_backward2_m1_last(*, d, x):
        """Calculate the subroutine for the last vertical layer."""
        z = backward_m1_last(d)  # noqa (handled by stencil composition system)
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
        """Declare name."""
        return "periodic3"

    @staticmethod
    def periodic3_full(x, z, fact):
        """Calculate the subroutine."""
        return x - fact * z
