from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("FullXCM2_cython", ["FullXCM2_cython.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

setup(
    ext_modules=cythonize("FullXCM2_cython.pyx"),
    include_dirs=[numpy.get_include()]
)
