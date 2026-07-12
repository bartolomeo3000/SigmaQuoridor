"""Build the quoridor_cpp extension:  python setup_cpp.py build_ext --inplace"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="quoridor_cpp",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "quoridor_cpp",
            ["cpp/bindings.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": build_ext},
)
