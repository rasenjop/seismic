python setup.py build_ext --inplace
rm -rf build
#on MAC instal gcc-9.1 con homebrew

# Set DYLIB_LIBRARY_PATH=/usr/local/Cellar/gcc/8.2.0/lib/gcc/8 python test.py
# CC=g++-9 python par_setup.py build_ext --inplace
python par_setup.py build_ext --inplace
