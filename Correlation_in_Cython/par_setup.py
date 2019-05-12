from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()
ext = Extension(
    name="par_correlate_cython_wrapper",
    sources=["par_correlate_cython_wrapper.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp']
)

setup(ext_modules=cythonize(ext))
