"""Hand-implement stencils for step-wise debugging while porting."""
import numpy

from gt4py_benchmarks.stencils.tooling import StorageBuilder


class HorizontalDiffusion:
    """Horizontal diffusion integration stepper, hand-implemented stencil for debugging."""

    SCALAR_T = numpy.float64

    def __init__(self, *, backend, dspace, coeff):
        """
        Construct from spacial resolution and diffusion coefficient.

        `backend` arg is just for compatibility with stencils.
        """
        self.dx = self.SCALAR_T(dspace[0])
        self.dy = self.SCALAR_T(dspace[1])
        self.coeff = coeff
        self.weight = (-1.0 / 90.0, 5.0 / 36.0, -49.0 / 36.0, 49.0 / 36.0, -5.0 / 36.0, 1.0 / 90.0)

    def __call__(self, out, inp, *, dt):
        """Run the calculation."""
        flx_x0 = numpy.sum(self.weight[i] * inp[i : (-6 + i)] for i in range(0, 5)) / self.dx
        flx_x1 = numpy.sum(self.weight[i] * inp[(i + 1) : (-5 + i)] for i in range(0, 5)) / self.dx
        flx_y0 = numpy.sum(self.weight[i] * inp[:, i : (-6 + i)] for i in range(0, 5)) / self.dy
        flx_y1 = (
            numpy.sum(self.weight[i] * inp[:, (i + 1) : (-5 + i)] for i in range(0, 5)) / self.dy
        )

        flx_x0 = numpy.where(flx_x0 * (inp[3:-3] - inp[2:-4]) < 1e-2, 0.0, flx_x0)
        flx_x1 = numpy.where(flx_x1 * (inp[4:-2] - inp[3:-3]) < 1e-2, 0.0, flx_x1)
        flx_y0 = numpy.where(flx_y0 * (inp[:, 3:-3] - inp[:, 2:-4]) < 1e-2, 0.0, flx_y0)
        flx_y1 = numpy.where(flx_y1 * (inp[:, 4:-2] - inp[:, 3:-3]) < 1e-2, 0.0, flx_y1)

        out[3:-3, 3:-3] = inp[3:-3, 3:-3] + (
            self.coeff
            * dt
            * (((flx_x1 - flx_x0)[:, 3:-3] / self.dx) + ((flx_y1 - flx_y0)[3:-3] / self.dy))
        )

    def run(self, out, inp, *, dt, tmax):
        """Run one iteration step."""
        for t in numpy.arange(0, tmax + dt, dt):
            self(out, inp, dt=dt)
            tmp = out
            out = inp
            inp = tmp
        return out

    def storage_builder(self):
        """Create a preconfigured :class:`StorageBuilder` instance."""
        return StorageBuilder().backend("numpy").dtype(self.SCALAR_T)


class VerticalDiffusion:
    """Vertical diffusion integration stepper, hand-implemented stencil for debugging."""

    SCALAR_T = numpy.float64

    def __init__(self, *, backend, dspace, coeff):
        """
        Construct from spacial resolution and diffusion coefficient.

        `backend` arg is just for compatibility with stencils.
        """
        self.dz = self.SCALAR_T(dspace[2])
        self.coeff = coeff

    def forward_upper(self, a, b, c, d, *, k):
        """Apply first forward stage to middle and last vertical layers."""
        c[:, :, k] = c[:, :, k] / (b[:, :, k] - c[:, :, k - 1] * a[:, :, k])
        d[:, :, k] = (d[:, :, k] - a[:, :, k] * d[:, :, k - 1]) / (
            b[:, :, k] - c[:, :, k - 1] * a[:, :, k]
        )

    def forward_first(self, a, b, c, d, *, k):
        """Apply first forward stage to first vertical layer."""
        c[:, :, k] = c[:, :, k] / b[:, :, k]
        d[:, :, k] = d[:, :, k] / b[:, :, k]

    def periodic_forward1_first(self, a, b, c, d, alpha, beta, gamma, *, k):
        """Apply first forward periodic boundary condition stage to first vertical layer."""
        b[:, :, k] -= gamma
        self.forward_first(a, b, c, d, k=k)

    def periodic_forward1_mid(self, a, b, c, d, alpha, beta, gamma, *, k):
        """Apply first forward periodic boundary condition stage to middle vertical layers."""
        self.forward_upper(a, b, c, d, k=k)

    def periodic_forward1_last(self, a, b, c, d, alpha, beta, gamma, *, k):
        """Apply first forward periodic boundary condition stage to last vertical layer."""
        b[:, :, k] -= alpha * beta / gamma
        self.forward_upper(a, b, c, d, k=k)

    def backward_lower(self, out, c, d, *, k):
        """Apply the backwards stage to the first and middle layers."""
        out[:, :, k] = d[:, :, k] - c[:, :, k] * out[:, :, k + 1]

    def backward_last(self, out, c, d, *, k):
        """Apply the backwards stage to the last vertical layer."""
        out[:, :, k] = d[:, :, k]

    def periodic_forward2_first(self, a, b, c, d, alpha, gamma, *, k):
        """Apply second forward periodic boundary condition stage to first vertical layer."""
        d[:, :, k] = gamma
        self.forward_first(a, b, c, d, k=k)

    def periodic_forward2_mid(self, a, b, c, d, alpha, gamma, *, k):
        """Apply second forward periodic boundary condition stage to middle vertical layers."""
        d[:, :, k] = 0.0
        self.forward_upper(a, b, c, d, k=k)

    def periodic_forward2_last(self, a, b, c, d, alpha, gamma, *, k):
        """Apply second forward periodic boundary condition stage to last vertical layer."""
        d[:, :, k] = alpha
        self.forward_upper(a, b, c, d, k=k)

    def periodic_backward2_first(self, z, c, d, x, beta, gamma, fact, z_top, x_top, *, k):
        """Apply second backward periodic boundary condition stage to first vertical layer."""
        self.backward_lower(z, c, d, k=k)
        fact[:, :] = (x[:, :, k] + beta * x_top / gamma) / (
            1.0 + z[:, :, k] + beta * z_top / gamma
        )

    def periodic_backward2_mid(self, z, c, d, x, beta, gamma, fact, z_top, x_top, *, k):
        """Apply second backward periodic boundary condition stage to middle vertical layers."""
        self.backward_lower(z, c, d, k=k)

    def periodic_backward2_last(self, z, c, d, x, beta, gamma, fact, z_top, x_top, *, k):
        """Apply second backward periodic boundary condition stage to last vertical layer."""
        self.backward_last(z, c, d, k=k)

        z_top[:, :] = z[:, :, k]
        x_top[:, :] = x[:, :, k]

    def periodic3(self, data_out, x, z, fact):
        """Apply periodic boundary conditions in third stage."""
        data_out[:, :, :] = x - fact * z

    def diffusion_w_forward1_all(self, alpha, beta, gamma, a, b, c, d, data, data_tmp, dt):
        """Initialize temporaries for first forward diffusion stage."""
        a[:, :, :] = c[:, :, :] = -self.coeff / (2.0 * self.dz * self.dz)
        b[:, :, :] = 1.0 / dt - a - c

    def diffusion_w_forward1_first(self, alpha, beta, gamma, a, b, c, d, data, data_tmp, dt, *, k):
        """Apply first forward diffusion stage to first vertical layer."""
        d[:, :, k] = 1.0 / dt * data[:, :, k] + 0.5 * self.coeff * (
            data_tmp - 2.0 * data[:, :, k] + data[:, :, k + 1]
        ) / (self.dz * self.dz)

        alpha[:, :] = beta[:, :] = -self.coeff / (2.0 * self.dz * self.dz)
        gamma[:, :] = -b[:, :, k]

        self.periodic_forward1_first(a, b, c, d, alpha, beta, gamma, k=k)

        data_tmp[:, :] = data[:, :, k]

    def diffusion_w_forward1_mid(self, alpha, beta, gamma, a, b, c, d, data, data_tmp, dt, *, k):
        """Apply first forward diffusion stage to middle vertical layers."""
        d[:, :, k] = 1.0 / dt * data[:, :, k] + 0.5 * self.coeff * (
            data[:, :, k - 1] - 2.0 * data[:, :, k] + data[:, :, k + 1]
        ) / (self.dz * self.dz)

        self.periodic_forward1_mid(a, b, c, d, alpha, beta, gamma, k=k)

    def diffusion_w_forward1_last(self, alpha, beta, gamma, a, b, c, d, data, data_tmp, dt, *, k):
        """Apply first forward diffusion stage to last vertical layer."""
        d[:, :, k] = 1.0 / dt * data[:, :, k] + 0.5 * self.coeff * (
            data[:, :, k - 1] - 2.0 * data[:, :, k] + data_tmp
        ) / (self.dz * self.dz)

        self.periodic_forward1_last(a, b, c, d, alpha, beta, gamma, k=k)

    def stage_diffusion_w0(self, data, data_top):
        """Copy the last vertical layer of `data` into `data_top`."""
        data_top[:, :] = data[:, :, -1]

    def stage_diffusion_w_forward1(self, alpha, beta, gamma, a, b, c, d, data, data_tmp, dt):
        """Apply `diffusion_w_forward1` across vertical layers."""
        self.diffusion_w_forward1_all(alpha, beta, gamma, a, b, c, d, data, data_tmp, dt)
        for k in range(data.shape[2]):
            if k == 0:
                self.diffusion_w_forward1_first(
                    alpha, beta, gamma, a, b, c, d, data, data_tmp, dt, k=k
                )
            elif k == data.shape[2] - 1:
                self.diffusion_w_forward1_last(
                    alpha, beta, gamma, a, b, c, d, data, data_tmp, dt, k=k
                )
            else:
                self.diffusion_w_forward1_mid(
                    alpha, beta, gamma, a, b, c, d, data, data_tmp, dt, k=k
                )

    def stage_diffusion_w_backward1(self, x, c, d):
        """Apply `backward` across vertical layers."""
        for k in range(x.shape[2] - 1, -1, -1):
            if k == x.shape[2] - 1:
                self.backward_last(x, c, d, k=k)
            else:
                self.backward_lower(x, c, d, k=k)

    def stage_diffusion_w_forward2(self, a, b, c, d, alpha, gamma):
        """Apply `periodic_forward2` across vertical layers."""
        for k in range(a.shape[2]):
            if k == 0:
                self.periodic_forward2_first(a, b, c, d, alpha, gamma, k=k)
            elif k == a.shape[2] - 1:
                self.periodic_forward2_last(a, b, c, d, alpha, gamma, k=k)
            else:
                self.periodic_forward2_mid(a, b, c, d, alpha, gamma, k=k)

    def stage_diffusion_w_backward2(self, z, c, d, x, beta, gamma, fact, z_top, x_top):
        """Apply `periodic_backward2` across vertical layers."""
        for k in range(x.shape[2] - 1, -1, -1):
            if k == x.shape[2] - 1:
                self.periodic_backward2_last(z, c, d, x, beta, gamma, fact, z_top, x_top, k=k)
            elif k == 0:
                self.periodic_backward2_first(z, c, d, x, beta, gamma, fact, z_top, x_top, k=k)
            else:
                self.periodic_backward2_mid(z, c, d, x, beta, gamma, fact, z_top, x_top, k=k)

    def stage_diffusion_w3(
        self,
        out: numpy.array,
        x: numpy.array,
        z: numpy.array,
        fact: numpy.array,
        inp: numpy.array,
        dt: float,
    ):
        """Call :method:`periodic3` `(out, x, z, fact)`."""
        self.periodic3(out, x, z, fact)

    def __call__(self, out: numpy.array, data: numpy.array, *, dt: float):
        """Run the calculation."""
        # initializers
        a = numpy.zeros(data.shape, dtype=self.SCALAR_T)
        b = numpy.zeros(data.shape, dtype=self.SCALAR_T)
        c = numpy.zeros(data.shape, dtype=self.SCALAR_T)
        d = numpy.zeros(data.shape, dtype=self.SCALAR_T)
        x = numpy.zeros(data.shape, dtype=self.SCALAR_T)
        z = numpy.zeros(data.shape, dtype=self.SCALAR_T)
        data_tmp = numpy.zeros(data.shape[:2], dtype=self.SCALAR_T)
        alpha = numpy.zeros(data.shape[:2], dtype=self.SCALAR_T)
        beta = numpy.zeros(data.shape[:2], dtype=self.SCALAR_T)
        gamma = numpy.zeros(data.shape[:2], dtype=self.SCALAR_T)
        z_top = numpy.zeros(data.shape[:2], dtype=self.SCALAR_T)
        x_top = numpy.zeros(data.shape[:2], dtype=self.SCALAR_T)
        fact = numpy.zeros(data.shape[:2], dtype=self.SCALAR_T)

        # Stage parallel: diffusion_w0
        self.stage_diffusion_w0(data, data_tmp)

        # Stage forward: diffusion_w1_forward
        self.stage_diffusion_w_forward1(alpha, beta, gamma, a, b, c, d, data, data_tmp, dt)

        # Stage backward: periodic_backward1
        self.stage_diffusion_w_backward1(x, c, d)

        # Stage forward: periodic_forward2
        self.stage_diffusion_w_forward2(a, b, c, d, alpha, gamma)

        # Stage backward: periodic_backward2
        self.stage_diffusion_w_backward2(z, c, d, x, beta, gamma, fact, z_top, x_top)

        # Stage parallel: periodic3
        self.stage_diffusion_w3(out, x, z, fact, data, dt)

    def storage_builder(self):
        """Create a preconfigured :class:`StorageBuilder` instance."""
        return StorageBuilder().backend("numpy").dtype(self.SCALAR_T)
