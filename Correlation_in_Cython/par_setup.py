from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

from distutils import sysconfig

# me make damn sure, that disutils does not mess with our
# build process

sysconfig.get_config_vars()['CFLAGS'] = '-Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/anaconda3/include -I/anaconda3/include/python3.6m'
sysconfig.get_config_vars()['OPT'] = ''
sysconfig.get_config_vars()['PY_CFLAGS'] = ''
sysconfig.get_config_vars()['PY_CORE_CFLAGS'] = ''
sysconfig.get_config_vars()['CC'] = 'gcc-9'
sysconfig.get_config_vars()['CXX'] = 'g++-9'
sysconfig.get_config_vars()['BASECFLAGS'] = ''
sysconfig.get_config_vars()['CCSHARED'] = '-fPIC'
sysconfig.get_config_vars()['LDSHARED'] = 'gcc-9 -bundle -undefined dynamic_lookup -L/anaconda3/lib -arch x86_64'
sysconfig.get_config_vars()['CPP'] = ''
sysconfig.get_config_vars()['CPPFLAGS'] = ''
sysconfig.get_config_vars()['BLDSHARED'] = ''
sysconfig.get_config_vars()['CONFIGURE_LDFLAGS'] = ''
sysconfig.get_config_vars()['LDFLAGS'] = ''
sysconfig.get_config_vars()['PY_LDFLAGS'] = ''

ext = Extension(
    name="par_correlate_cython_wrapper",
    sources=["par_correlate_cython_wrapper.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3', '-fopenmp'],
    extra_link_args=['-fopenmp','-Wl,-rpath,/usr/local/Cellar/gcc/9.1.0/lib/gcc/9']
)

setup(ext_modules=cythonize(ext))
