"""Build the quoridor_cpp extension:  python setup_cpp.py build_ext --inplace"""
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# -O3 is a GCC/Clang flag; MSVC (Windows) doesn't recognize it (silently
# ignored with a D9002 warning, or errors), which would leave the C++ engine
# built WITHOUT optimization -- catastrophic for the BFS/MCTS-heavy engine
# code even though it has no effect on PyTorch/CUDA's own eval throughput.
extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]

setup(
    name="quoridor_cpp",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "quoridor_cpp",
            ["cpp/bindings.cpp"],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
