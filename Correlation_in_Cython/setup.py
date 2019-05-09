from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize('correlate_cython_wrapper.pyx'))