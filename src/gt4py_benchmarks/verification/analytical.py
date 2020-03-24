import numpy


def horizontal_diffusion(coordinates: tuple, diffusion_coeff: float, time: float):
    """Calculate horizontal diffusion reference value at given coordinates and time with given diffusion coefficient."""
    x, y, _ = coordinates
    return numpy.sin(x) * numpy.cos(y) * numpy.exp(-2 * diffusion_coeff * time)


def vertical_diffusion(coordinates: tuple, diffusion_coeff: float, time: float):
    """Calculate vertical diffusion reference value at given coordinates and time with given diffusion coefficient."""
    z = coordinates[2]
    return numpy.cos(z) * numpy.exp(-diffusion_coeff * time)


def full_diffusion(coordinates: tuple, diffusion_coeff: float, time: float):
    """Calculate full diffusion reference value for given coordinates, time and diffusion coefficient."""
    x, y, z = coordinates
    return numpy.sin(x) * numpy.cos(y) * numpy.cos(z) * numpy.exp(-3 * diffusion_coeff * time)


## horizontal advection velocity: (5, -2, 0)
def horizontal_advection(coordinates: tuple, time: float):
    """Calculate horizontal advection reference value at given coordinates and time."""
    x, y, z = coordinates
    return numpy.sin(x - 5 * t) * numpy.cos(y + 2 * time) * numpy.cos(z)


## vertical advection velocity: (0, 0, 3)
def vertical_advection(coordinates: tuple, time: float):
    """Calculate vertical advection reference value at given coordinates and time."""
    x, y, z = coordinates
    return numpy.sin(x) * numpy.cos(y) * numpy.cos(z - 3 * time)


## full advection velocity: (5, -2, 3)
def full_advection(coordinates: tuple, time: float):
    """Calculate full advection reference value at given coordinates and time."""
    x, y, z = coordinates
    return numpy.sin(x - 5 * time) * numpy.cos(y + 2 * time) * numpy.cos(z - 3 * time)


def advection_diffusion(coordinates: tuple, diffusion_coeff: float, time: float):
    x, y, z = coordinates
    a = numpy.sqrt(2) / 2
    return -numpy.sin(x) * numpy.sin(a * (y - z)) * numpy.exp(-2 * diffusion_coeff * time)
