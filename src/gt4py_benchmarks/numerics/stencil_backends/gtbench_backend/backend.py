import importlib
import multiprocessing
import pathlib
import subprocess
import sys
import sysconfig

import typing_extensions

from .. import base
from ....constants import HALO


GTBENCH_PATH = pathlib.Path(__file__).parent / "gtbench"
BUILD_PATH = pathlib.Path(".gtbench_cache")


class GTBenchStencilBackend(base.StencilBackend):
    gtbench_backend: typing_extensions.Literal["cpu_ifirst", "cpu_kfirst", "gpu"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._module_name = f"gtbench_{self.gtbench_backend}_{self.dtype}"
        self._build_dir = BUILD_PATH / self._module_name
        self._library = self._build_dir / "python" / f"{self._module_name}.so"
        if not self._library.exists():
            self._configure_gtbench()
            self._build_gtbench()
        self._gtbench = self._load_gtbench()
        assert self._gtbench.halo == HALO
        assert self._gtbench.dtype == self.dtype
        assert self._gtbench.backend == self.gtbench_backend

    def _configure_gtbench(self):
        self._build_dir.mkdir(parents=True, exist_ok=True)
        cpp_dtype = "float" if self.dtype == "float32" else "double"
        command = [
            "cmake",
            str(GTBENCH_PATH.absolute()),
            f"-DCMAKE_INSTALL_PREFIX={self._build_dir.absolute()}/install",
            "-DGTBENCH_PYTHON_BINDINGS=ON",
            "-DGTBENCH_RUNTIME=single_node",
            f"-DGTBENCH_BACKEND={self.gtbench_backend}",
            f"-DGTBENCH_FLOAT={cpp_dtype}",
            f"-DGTBENCH_PYTHON_MODULE_NAME={self._module_name}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
        ]
        subprocess.run(command, check=True, cwd=self._build_dir)

    def _build_gtbench(self):
        command = ["make", "-j", str(multiprocessing.cpu_count())]
        subprocess.run(command, check=True, cwd=self._build_dir)

    def _load_gtbench(self):
        spec = importlib.util.spec_from_file_location(self._module_name, self._library)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def storage_from_array(self, array):
        return self._gtbench.storage_from_array(array)

    def array_from_storage(self, storage):
        return self._gtbench.array_from_storage(storage)

    def synchronize(self):
        import cupy as cp
        cp.cuda.Device().synchronize()

    def hdiff_stepper(self, diffusion_coeff):
        return self._gtbench.hdiff_stepper(diffusion_coeff)

    def vdiff_stepper(self, diffusion_coeff):
        return self._gtbench.vdiff_stepper(diffusion_coeff)

    def diff_stepper(self, diffusion_coeff):
        return self._gtbench.diff_stepper(diffusion_coeff)

    def hadv_stepper(self):
        return self._gtbench.hadv_stepper()

    def vadv_stepper(self):
        return self._gtbench.vadv_stepper()

    def rkadv_stepper(self):
        return self._gtbench.rkadv_stepper()

    def advdiff_stepper(self, diffusion_coeff):
        return self._gtbench.advdiff_stepper(diffusion_coeff)
