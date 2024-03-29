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
        return np.sin(x - 0.2 * t) * np.cos(y + 0.3 * t) * np.cos(z)

    def u(x, y, z, t):
        return np.full_like(x, 0.2)

    def v(x, y, z, t):
        return np.full_like(x, -0.3)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data, u=u, v=v)


def vertical_advection():
    def data(x, y, z, t):
        return np.sin(x) * np.cos(y) * np.cos(z - 0.6 * t)

    def w(x, y, z, t):
        return np.full_like(x, 0.6)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data, w=w)


def full_advection():
    def data(x, y, z, t):
        return np.sin(x - 0.1 * t) * np.cos(y + 0.2 * t) * np.cos(z - 0.3 * t)

    def u(x, y, z, t):
        return np.full_like(x, 0.1)

    def v(x, y, z, t):
        return np.full_like(x, -0.2)

    def w(x, y, z, t):
        return np.full_like(x, 0.3)

    return AnalyticalSolution(domain=(4 * np.pi, 4 * np.pi, 4 * np.pi), data=data, u=u, v=v, w=w)


def advection_diffusion(diffusion_coeff):
    a = np.sqrt(2) / 2

    def data(x, y, z, t):
        return -np.sin(x) * np.sin(a * (y - z)) * np.exp(-2 * diffusion_coeff * t)

    def u(x, y, z, t):
        return -np.sin(x) * np.cos(a * (y - z)) * 0.1

    def v(x, y, z, t):
        return a * np.cos(x) * np.sin(a * (y - z)) * 0.1

    def w(x, y, z, t):
        return -a * np.cos(x) * np.sin(a * (y - z)) * 0.1

    return AnalyticalSolution(
        domain=(2 * np.pi, 2 * np.pi * np.sqrt(2), 2 * np.pi * np.sqrt(2)),
        data=data,
        u=u,
        v=v,
        w=w,
    )


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
            analytical.domain[0] * repeats[0],
            analytical.domain[1] * repeats[1],
            analytical.domain[2] * repeats[2],
        ),
        data=remap(analytical.data),
        u=remap(analytical.u),
        v=remap(analytical.v),
        w=remap(analytical.w),
    )
