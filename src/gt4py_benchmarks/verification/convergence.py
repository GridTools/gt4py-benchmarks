import typing

import numpy as np


class OrderVerificationResult(typing.NamedTuple):
    ns: np.ndarray
    errors: np.ndarray
    orders: np.ndarray

    def __str__(self):
        s = "{:15} {:15} {:15}\n".format("Resolution", "Error", "Order")
        for n, e, o in zip(self.ns, self.errors, self.orders):
            s += f"{n:<15} {e:<15.5e}"
            if not np.isnan(o):
                s += f" {o:<15.2f}"
            s += "\n"
        return s


def order_verification(f, n_min, n_max):
    ns = n_min * 2 ** np.arange(int(np.log2(n_max / n_min)) + 1)
    errors = np.array([f(n) for n in ns])
    orders = np.empty_like(errors)
    orders[0] = np.nan
    orders[1:] = np.log2(errors[:-1] / errors[1:])

    return OrderVerificationResult(ns, errors, orders)
