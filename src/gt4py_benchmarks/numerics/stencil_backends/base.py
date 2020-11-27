import abc

import numpy as np
import pydantic

from ...constants import HALO
from ...utils import registry


@registry.subclass_registry
class StencilBackend(pydantic.BaseModel, abc.ABC):
    dtype: str

    @abc.abstractmethod
    def storage_from_array(self, array):
        pass

    def array_from_storage(self, storage):
        return np.asarray(storage)

    def hdiff_stencil(self, resolution, delta, diffusion_coeff):
        raise NotImplementedError()

    def vdiff_stencil(self, resolution, delta, diffusion_coeff):
        raise NotImplementedError()

    def hadv_stencil(self, resolution, delta):
        raise NotImplementedError()

    def vadv_stencil(self, resolution, delta):
        raise NotImplementedError()

    def hdiff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            hdiff = self.hdiff_stencil(state.resolution, state.delta, diffusion_coeff)

            def step(state, dt):
                exchange(state.data)
                hdiff(state.data1, state.data, dt)
                state.data, state.data1 = state.data1, state.data

            return step

        return stepper

    def vdiff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            vdiff = self.vdiff_stencil(state.resolution, state.delta, diffusion_coeff)

            def step(state, dt):
                vdiff(state.data1, state.data, dt)
                state.data, state.data1 = state.data1, state.data

            return step

        return stepper

    def diff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            hdiff = self.hdiff_stencil(state.resolution, state.delta, diffusion_coeff)
            vdiff = self.vdiff_stencil(state.resolution, state.delta, diffusion_coeff)

            def step(state, dt):
                exchange(state.data)
                hdiff(state.data1, state.data, dt)
                vdiff(state.data, state.data1, dt)

            return step

        return stepper

    def hadv_stepper(self):
        def stepper(state, exchange):
            hadv = self.hadv_stencil(state.resolution, state.delta)

            def step(state, dt):
                exchange(state.data)
                hadv(state.data1, state.data, state.data, state.u, state.v, dt)
                state.data, state.data1 = state.data1, state.data

            return step

        return stepper

    def vadv_stepper(self):
        def stepper(state, exchange):
            vadv = self.vadv_stencil(state.resolution, state.delta)

            def step(state, dt):
                exchange(state.data)
                vadv(state.data1, state.data, state.data, state.w, dt)
                state.data, state.data1 = state.data1, state.data

            return step

        return stepper

    def rkadv_stepper(self):
        def stepper(state, exchange):
            hadv = self.hadv_stencil(state.resolution, state.delta)
            vadv = self.vadv_stencil(state.resolution, state.delta)

            def step(state, dt):
                exchange(state.data)
                hadv(state.data1, state.data, state.data, state.u, state.v, dt / 3)
                vadv(state.data1, state.data, state.data1, state.w, dt / 3)
                exchange(state.data1)
                hadv(state.data2, state.data1, state.data, state.u, state.v, dt / 2)
                vadv(state.data2, state.data1, state.data2, state.w, dt / 2)
                exchange(state.data2)
                hadv(state.data1, state.data2, state.data, state.u, state.v, dt)
                vadv(state.data, state.data2, state.data1, state.w, dt)

            return step

        return stepper

    def advdiff_stepper(self, diffusion_coeff):
        def stepper(state, exchange):
            hdiff = self.hdiff_stepper(diffusion_coeff)(state, exchange)
            vdiff = self.vdiff_stepper(diffusion_coeff)(state, exchange)
            rkadv = self.rkadv_stepper()(state, exchange)

            def step(state, dt):
                hdiff(state, dt)
                rkadv(state, dt)
                vdiff(state, dt)

            return step

        return stepper

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
