python setup.py build_ext --inplace
rm -rf build
#on MAC instal gcc-9.1 con homebrew

CC=g++-9 python par_setup.py build_ext --inplace
