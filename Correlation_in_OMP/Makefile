
lib: correlation_c.cpp
	g++-8 -g -O3 -c -lfftw3_omp -lfftw3f -lm correlation_c.cpp
	g++-8 -o correlation_c.dylib -fPIC -shared -lfftw3f -lm -O3 -fopenmp correlation_c.o

clean:
	rm correlation_c.dylib correlation_c.o
