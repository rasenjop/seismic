#!bin/bash

cd build
make
cd ..
python test_AF.py
