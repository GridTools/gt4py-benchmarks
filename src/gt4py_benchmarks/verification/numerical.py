import numpy
import mdebug

from gt4py_benchmarks.stencils.tooling import StorageBuilder


class shape_analyzer(mdebug.analyzer.analyzer_interface):
    def analyze_source(self, expr_string):
        self._storage["source"] = expr_string

    def analyze_value(self, value):
        self._storage["shape"] = "has no shape."
        if hasattr(value, "shape"):
            self._storage["shape"] = value.shape

    def display_analysis(self):
        print(f'\t{self._storage["source"]}.shape = {self._storage["shape"]}')


DBG = mdebug.mdebug(analyzer=shape_analyzer(), novalue=True)


class HorizontalDiffusion:
    SCALAR_T = numpy.float64

    def __init__(self, *, backend, dspace, coeff):
        self.dx = self.SCALAR_T(dspace[0])
        self.dy = self.SCALAR_T(dspace[1])
        self.coeff = coeff
        self.weight = (-1.0 / 90.0, 5.0 / 36.0, -49.0 / 36.0, 49.0 / 36.0, -5.0 / 36.0, 1.0 / 90.0)

    def __call__(self, out, inp, *, dt):
        flx_x0 = numpy.sum(self.weight[i] * inp[i : (-6 + i)] for i in range(0, 5)) / self.dx
        flx_x1 = numpy.sum(self.weight[i] * inp[(i + 1) : (-5 + i)] for i in range(0, 5)) / self.dx
        flx_y0 = numpy.sum(self.weight[i] * inp[:, i : (-6 + i)] for i in range(0, 5)) / self.dy
        flx_y1 = (
            numpy.sum(self.weight[i] * inp[:, (i + 1) : (-5 + i)] for i in range(0, 5)) / self.dy
        )

        flx_x0 *= flx_x0
        flx_x1 *= flx_x1
        flx_y0 *= flx_y0
        flx_y1 *= flx_y1

        flx_x0 = numpy.where((inp[3:-3] - inp[2:-4]) < 0.0, 0.0, flx_x0)
        flx_x1 = numpy.where((inp[4:-2] - inp[3:-3]) < 0.0, 0.0, flx_x1)
        flx_y0 = numpy.where((inp[:, 3:-3] - inp[:, 2:-4]) < 0.0, 0.0, flx_y0)
        flx_y1 = numpy.where((inp[:, 4:-2] - inp[:, 3:-3]) < 0.0, 0.0, flx_y1)

        out[3:-3, 3:-3] = inp[3:-3, 3:-3] + (
            self.coeff
            * dt
            * (((flx_x1 - flx_x0)[:, 3:-3] / self.dx) + ((flx_y1 - flx_y0)[3:-3] / self.dy))
        )

    def storage_builder(self):
        return StorageBuilder().backend("numpy").dtype(self.SCALAR_T)
