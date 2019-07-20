#CFLAGS = -O -fopenmp -g -I$(EXTRAE_HOME)/include
#CLIBS = -L$(EXTRAE_HOME)/lib -lomptrace

lib: correlation_multicuda.cu
	nvcc --shared -lcufft -lgomp -o correlation_multicuda.so correlation_multicuda.cu --compiler-options \
	'-fPIC' '-O3' '-std=c++11' '-L/gpfs/apps/NVIDIA/FFTW/3.3/lib' '-lfftw3f' '-lm' '-Xcompiler="-pthread"' \
	'-Xcompiler="-fopenmp"' '${BSCGPU_CFLAGS}'
##'-I/gpfs/apps/NVIDIA/CUDA/5.0/include'
##	g++ -g -O3 -fPIC -c -fopenmp ${BSCGPU_CFLAGS} correlation_c.cpp
##	g++ -o correlation_c.dylib -fPIC -shared -lfftw3 -lm -O3 -fopenmp correlation_c.o
##	g++ -o correlation_c.so -fPIC -shared -L/gpfs/apps/NVIDIA/FFTW/3.3/lib -lfftw3f -lm -O3 -fopenmp correlation_c.o
##  g++ -o correlation_c.so -fPIC -shared ${BSCGPU_LDFLAGS} -lm -O3 -fopenmp correlation_c.o

#lib: correlation_c.cpp
#	g++ -g -O3 -fPIC -c -fopenmp ${BSCGPU_CFLAGS} -I$(EXTRAE_HOME)/include correlation_c.cpp
#	g++ -g -o correlation_c.so -fPIC -shared -L/gpfs/apps/NVIDIA/FFTW/3.3/lib -lfftw3f -lm -L$(EXTRAE_HOME)/lib -lomptrace -O3 -fopenmp correlation_c.o

clean:
	rm correlation_multicuda.so
