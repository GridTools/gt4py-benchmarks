"""Simulation wrappers around time step stencils."""
import copy
import functools
import collections
import typing

import numpy

from gt4py_benchmarks.verification import analytical
from gt4py_benchmarks.stencils import diffusion, advection


class Simulation:
    """Wrap stencil and reference into a simulation to verify convergence."""

    def __init__(self, sim_spec: typing.Mapping, *, backend: str):
        """Construct from a simulation specification and the backend fixture."""
        self.domain = sim_spec.get("domain", analytical.DOMAIN)
        self.time_step = 1e-3
        self.max_time = 1e-2
        self.shape = sim_spec.get("shape", (16, 16, 16))
        self.backend_name = backend
        self.tolerance = sim_spec["tolerance"]
        dspace = numpy.array(analytical.DOMAIN, dtype=numpy.float64) / numpy.array(
            self.shape, dtype=numpy.float64
        )
        stencil_args = {
            "backend": self.backend_name,
            "dspace": dspace,
            "time_step": self.time_step,
        }
        stencil_args.update(sim_spec.get("extra-args", {}))
        self.extra_args = sim_spec.get("extra-args", {})
        self.stencil = sim_spec["stencil"](**stencil_args)
        self.reference = sim_spec["reference"]
        storage_b = self.stencil.storage_builder().default_origin(self.stencil.min_origin())

        self.data = storage_b.from_array(numpy.fromfunction(self.get_reference, shape=self.shape))
        self.data1 = copy.deepcopy(self.data)
        self._initial_state = copy.deepcopy(self.data)
        self._expected = numpy.fromfunction(
            functools.partial(self.get_reference, time=self.max_time), shape=self.shape
        )

    def run(self):
        """Run the simulation until `self.max_time`."""
        time = 0
        while time <= self.max_time:
            self.step()
            time += self.time_step

    def step(self):
        """Run a simulation step."""
        self.stencil(
            self.data1, self.data, dt=self.time_step,
        )
        self._swap_data()

    def __repr__(self):
        """Build a helpful string representation in case a test fails."""
        return (
            f"<Simulation: stencil = {self.stencil.name()} "
            f"@ {self.backend_name} vs. {self.reference.__name__}>"
        )

    def map_to_domain(self, i: int, j: int, k: int):
        """Map from IJK coordinates to XYZ."""
        return analytical.map_domain(i, j, k, resolution=self.shape, domain=self.domain)

    def get_reference(self, i: int, j: int, k: int, time: float = 0.0):
        """Get reference values at IJK grid points."""
        return self.reference(*self.map_to_domain(i, j, k), time=time, **self.extra_args)

    def _swap_data(self):
        """Swap input and output buffers after time step."""
        tmp = self.data
        self.data = self.data1
        self.data1 = tmp

    @property
    def expected(self):
        """Construct the reference values on the grid at `t=max_time`."""
        return self._expected[3:-3, 3:-3, 1:-1]

    @property
    def result(self):
        """Return the current result at `t=current_time`."""
        return self.data[3:-3, 3:-3, 1:-1]

    @property
    def initial(self):
        """Return the initial state."""
        return self._initial_state[3:-3, 3:-3, 1:-1]

    @property
    def change(self):
        """Return the absolute differences between initial and current state."""
        return numpy.abs(self.initial - self.result)

    @property
    def expected_change(self):
        """Return the absolute difference between the expected result and the initial state."""
        return numpy.abs(self.expected - self.initial)

    @property
    def errors(self):
        """Return the absolute differences between current and expected state."""
        return numpy.abs(self.expected - self.result)

    @property
    def rel_errors(self):
        """Return relative errors."""
        return numpy.divide(
            self.errors,
            self.expected,
            out=numpy.zeros(self.errors.shape),
            where=(numpy.abs(self.expected) < 1e-12),
        )


class RkAdvSimulation(Simulation):
    """Specialize for the full (runge-kutta) advection stencil."""

    def __init__(self, sim_spec: typing.Mapping, *, backend: str):
        """Construct from simulation specification and the backend name."""
        super().__init__(sim_spec, backend=backend)
        self.data2 = copy.deepcopy(self.data)

    def step(self):
        """Run the RK advection step."""
        self.stencil(self.data1, self.data, self.data, dt=self.time_step / 3)
        self.stencil(self.data2, self.data1, self.data, dt=self.time_step / 2)
        self.stencil(self.data, self.data2, self.data, dt=self.time_step)


class AdvDiffSimulation(RkAdvSimulation):
    """Specialize for the full advection-diffusion process."""

    def __init__(self, sim_spec: typing.Mapping, *, backend: str):
        """Initialize all stencils from the otherwise same spec."""
        super().__init__(
            collections.ChainMap({"stencil": advection.Full}, sim_spec), backend=backend
        )
        self.vdiff = Simulation(
            collections.ChainMap(
                {"stencil": diffusion.Vertical, "reference": analytical.vertical_diffusion},
                sim_spec,
            ),
            backend=backend,
        ).stencil
        self.hdiff = Simulation(
            collections.ChainMap(
                {"stencil": diffusion.Horizontal, "reference": analytical.horizontal_diffusion},
                sim_spec,
            ),
            backend=backend,
        ).stencil
        self.rkadv = self.stencil

    def step(self):
        """Time step vertical diffusion, full advection and horizontal diffusion."""
        self.vdiff(self.data1, self.data, dt=self.time_step)
        self._swap_data()

        self.stencil(self.data1, self.data, self.data, dt=self.time_step / 3)
        self.stencil(self.data2, self.data1, self.data, dt=self.time_step / 2)
        self.stencil(self.data, self.data2, self.data, dt=self.time_step)

        self.hdiff(self.data1, self.data, dt=self.time_step)
        self._swap_data()
