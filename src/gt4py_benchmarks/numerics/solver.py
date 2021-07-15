import dataclasses
import typing


@dataclasses.dataclass
class SolverState:
    resolution: typing.Tuple[int, int, int]
    delta: typing.Tuple[float, float, float]
    u: typing.Any
    v: typing.Any
    w: typing.Any
    data: typing.Any
    data1: typing.Any
    data2: typing.Any
