import abc

import numpy as np

from ...constants import HALO


class StencilBackend(abc.ABC):
    def __init__(self, *, dtype):
        self.dtype = np.dtype(dtype)

    @abc.abstractmethod
    def storage_from_array(self, array):
        pass

    def array_from_storage(self, storage):
        return np.asarray(storage)

    @abc.abstractmethod
    def hdiff_stencil(self, resolution, delta, diffusion_coeff):
        pass

    @abc.abstractmethod
    def vdiff_stencil(self, resolution, delta, diffusion_coeff):
        pass

    @abc.abstractmethod
    def hadv_stencil(self, resolution, delta):
        pass

    @abc.abstractmethod
    def vadv_stencil(self, resolution, delta):
        pass

    @abc.abstractmethod
    def rkadv_stencil(self, resolution, delta):
        pass

    def hdiff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            hdiff = self.hdiff_stencil(state.resolution, state.delta, diffusion_coeff)

            def step(state, dt):
                exchange(state.data[0])
                hdiff(state.data[1], state.data[0], dt)
                state.data[0], state.data[1] = state.data[1], state.data[0]

            return step

        return stepper

    def vdiff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            vdiff = self.vdiff_stencil(state.resolution, state.delta, diffusion_coeff)

            def step(state, dt):
                vdiff(state.data[1], state.data[0], dt)
                state.data[0], state.data[1] = state.data[1], state.data[0]

            return step

        return stepper

    def diff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            hdiff = self.hdiff_stencil(state.resolution, state.delta, diffusion_coeff)
            vdiff = self.vdiff_stencil(state.resolution, state.delta, diffusion_coeff)

            def step(state, dt):
                exchange(state.data[0])
                hdiff(state.data[1], state.data[0], dt)
                vdiff(state.data[0], state.data[1], dt)

            return step

        return stepper

    def hadv_stepper(self):
        def stepper(state, exchange):
            hadv = self.hadv_stencil(state.resolution, state.delta)

            def step(state, dt):
                exchange(state.data[0])
                hadv(state.data[1], state.data[0], state.u, state.v, dt)
                state.data[0], state.data[1] = state.data[1], state.data[0]

            return step

        return stepper

    def vadv_stepper(self):
        def stepper(state, exchange):
            vadv = self.vadv_stencil(state.resolution, state.delta)

            def step(state, dt):
                exchange(state.data[0])
                vadv(state.data[1], state.data[0], state.w, dt)
                state.data[0], state.data[1] = state.data[1], state.data[0]

            return step

        return stepper

    def rkadv_stepper(self):
        def stepper(state, exchange):
            rkadv = self.rkadv_stencil(state.resolution, state.delta)

            def step(state, dt):
                exchange(state.data[0])
                rkadv(
                    state.data[1], state.data[0], state.data[0], state.u, state.v, state.w, dt / 3
                )
                exchange(state.data[1])
                rkadv(
                    state.data[2], state.data[1], state.data[0], state.u, state.v, state.w, dt / 2
                )
                exchange(state.data[2])
                rkadv(state.data[0], state.data[2], state.data[0], state.u, state.v, state.w, dt)

            return step

        return stepper

    def advdiff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            hdiff = self.hdiff_stencil(state.resolution, state.delta, diffusion_coeff)
            vdiff = self.vdiff_stencil(state.resolution, state.delta, diffusion_coeff)
            rkadv = self.rkadv_stencil(state.resolution, state.delta)

            def step(state, dt):
                vdiff(state.data[1], state.data[0], dt)
                state.data[0], state.data[1] = state.data[1], state.data[0]

                exchange(state.data[0])
                rkadv(
                    state.data[1], state.data[0], state.data[0], state.u, state.v, state.w, dt / 3
                )
                exchange(state.data[1])
                rkadv(
                    state.data[2], state.data[1], state.data[0], state.u, state.v, state.w, dt / 2
                )
                exchange(state.data[2])
                rkadv(state.data[0], state.data[2], state.data[0], state.u, state.v, state.w, dt)

                exchange(state.data[0])
                hdiff(state.data[1], state.data[0], dt)
                state.data[0], state.data[1] = state.data[1], state.data[0]

            return step

        return stepper
