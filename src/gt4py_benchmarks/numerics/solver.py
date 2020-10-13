import typing

import numpy as np


class SolverState(typing.NamedTuple):
    resolution: typing.Tuple[int, int, int]
    delta: typing.Tuple[float, float, float]
    data: typing.List[np.ndarray]
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
