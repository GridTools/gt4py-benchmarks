"""Analytical reference functions"""

import functools
import typing

import numpy as np


class AnalyticalSolution(typing.NamedTuple):
    domain: typing.Tuple[float, float, float] = (4 * np.pi, 4 * np.pi, 4 * np.pi)
    data: typing.Callable[
        [float, float, float, float], np.array
    ] = lambda x, y, z, t: np.zeros_like(x)
    u: typing.Callable[[float, float, float, float], np.array] = lambda x, y, z, t: np.zeros_like(
        x
    )
    v: typing.Callable[[float, float, float, float], np.array] = lambda x, y, z, t: np.zeros_like(
        x
    )
    w: typing.Callable[[float, float, float, float], np.array] = lambda x, y, z, t: np.zeros_like(
        x
    )


def horizontal_diffusion(diffusion_coeff):
    def data(x, y, z, t):
        return np.sin(x) * np.cos(y) * np.exp(-2 * diffusion_coeff * t)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data)


def vertical_diffusion(diffusion_coeff):
    def data(x, y, z, t):
        return np.cos(z) * np.exp(-diffusion_coeff * t)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data)


def full_diffusion(diffusion_coeff):
    def data(x, y, z, t):
        return np.sin(x) * np.cos(y) * np.cos(z) * np.exp(-3 * diffusion_coeff * t)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data)


def horizontal_advection():
    def data(x, y, z, t):
        return np.sin(x - 5 * t) * np.cos(y + 2 * t) * np.cos(z)

    def u(x, y, z, t):
        return np.full_like(x, 5)

    def v(x, y, z, t):
        return np.full_like(x, -2)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data, u=u, v=v)


def vertical_advection():
    def data(x, y, z, t):
        return np.sin(x) * np.cos(y) * np.cos(z - 3 * t)

    def w(x, y, z, t):
        return np.full_like(x, 3)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data, w=w)


def full_advection():
    def data(x, y, z, t):
        return np.sin(x - 5 * t) * np.cos(y + 2 * t) * np.cos(z - 3 * t)

    def u(x, y, z, t):
        return np.full_like(x, 5)

    def v(x, y, z, t):
        return np.full_like(x, -2)

    def w(x, y, z, t):
        return np.full_like(x, 3)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data, u=u, v=v, w=w)


def repeat(analytical: AnalyticalSolution, repeats: typing.Tuple[int, int, int]):
    def remap(f):
        @functools.wraps(f)
        def remapped(x, y, z, t):
            return f(
                np.fmod(x, analytical.domain[0]),
                np.fmod(y, analytical.domain[1]),
                np.fmod(z, analytical.domain[2]),
                t,
            )

        return remapped

    return AnalyticalSolution(
        domain=(
            analytical.domain * repeats[0],
            analytical.domain * repeats[1],
            analytical.domain * repeats[2],
        ),
        data=remap(analytical.data),
        u=remap(analytical.u),
        v=remap(analytical.v),
        w=remap(analytical.w),
    )
