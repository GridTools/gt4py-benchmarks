import time

from ...constants import HALO
from . import base
from .. import discrete_analytical


class SingleNodeRuntime(base.Runtime):
    def solve(self, analytical, stepper, global_resolution, tmax, dt):
        exact = discrete_analytical.discretize(
            analytical, global_resolution, global_resolution, (0, 0, 0)
        )
        state = self.init_state(exact)

        def exchange(field):
            field[:HALO, :, :] = field[-2 * HALO : -HALO, :, :]
            field[-HALO:, :, :] = field[HALO : 2 * HALO, :, :]
            field[:, :HALO, :] = field[:, -2 * HALO : -HALO, :]
            field[:, -HALO:, :] = field[:, HALO : 2 * HALO, :]

        step = stepper(state, exchange)

        if tmax > 0:
            step(state, dt)

        start = time.perf_counter()

        t = dt
        while t < tmax - dt / 2:
            step(state, dt)
            t += dt

        end = time.perf_counter()

        error = self.compute_error(state, exact, t)
        return base.SolveResult(error=error, time=end - start)
