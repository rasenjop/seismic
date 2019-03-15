#!bin/bash


g++-8 -g -std=c++11 -pthread -O3 -c correlation_pthreads.cpp 
g++-8 -o correlation_pthreads.dylib -fPIC -shared -lfftw3 -lm -pthread -O3 correlation_pthreads.o
