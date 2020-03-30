import numpy


DOMAIN = tuple([4 * numpy.pi] * 3)


def map_domain(
    i: float,
    j: float,
    k: float,
    *,
    resolution: tuple,
    domain: tuple,
    halo: float = 3,
    offset: tuple = (0, 0),
):
    x = (i - halo + offset[0]) * domain[0] / resolution[0]
    y = (j - halo + offset[1]) * domain[1] / resolution[1]
    z = k * domain[2] / resolution[2]
    return x, y, z


def horizontal_diffusion(x: float, y: float, z: float, *, diffusion_coeff: float, time: float):
    """Calculate horizontal diffusion reference value at given coordinates and time with given diffusion coefficient."""
    return numpy.sin(x) * numpy.cos(y) * numpy.exp(-2 * diffusion_coeff * time)


def vertical_diffusion(x: float, y: float, z: float, *, diffusion_coeff: float, time: float):
    """Calculate vertical diffusion reference value at given coordinates and time with given diffusion coefficient."""
    del x, y  ## not needed but kept for unity of interface
    return numpy.cos(z) * numpy.exp(-diffusion_coeff * time)


def full_diffusion(x: float, y: float, z: float, *, diffusion_coeff: float, time: float):
    """Calculate full diffusion reference value for given coordinates, time and diffusion coefficient."""
    return numpy.sin(x) * numpy.cos(y) * numpy.cos(z) * numpy.exp(-3 * diffusion_coeff * time)


## horizontal advection velocity: (5, -2, 0)
def horizontal_advection(x: float, y: float, z: float, *, time: float):
    """Calculate horizontal advection reference value at given coordinates and time."""
    return numpy.sin(x - 5 * t) * numpy.cos(y + 2 * time) * numpy.cos(z)


## vertical advection velocity: (0, 0, 3)
def vertical_advection(x: float, y: float, z: float, *, time: float):
    """Calculate vertical advection reference value at given coordinates and time."""
    return numpy.sin(x) * numpy.cos(y) * numpy.cos(z - 3 * time)


## full advection velocity: (5, -2, 3)
def full_advection(x: float, y: float, z: float, *, time: float):
    """Calculate full advection reference value at given coordinates and time."""
    return numpy.sin(x - 5 * time) * numpy.cos(y + 2 * time) * numpy.cos(z - 3 * time)


def advection_diffusion(x: float, y: float, z: float, *, diffusion_coeff: float, time: float):
    a = numpy.sqrt(2) / 2
    return -numpy.sin(x) * numpy.sin(a * (y - z)) * numpy.exp(-2 * diffusion_coeff * time)
