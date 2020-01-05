# Seismic
Seismic application Master's Thesis (Francisco López Sánchez, advisors: Dr. Thomas Grass, Prof. Dr. rer. nat. Rainer Leupers, and Prof. Rafael Asenjo)

This repository contains different optimizations of a Python code ("original" directory) developed from Andrew Bell (School of GeoSciences at the University of Edinburgh). This application was intended to process data captured by sensors which recorded seismic activity on the Galapagos Islands.

Thanks to Cython and Ctypes we keep the Python interface but optimize the computationally expensive part of the algorithm in C++ for parallel and multi-GPU based architectures. Summarizing, the best parallel CPU implementation is 6,121x faster than the python baseline. Using 4 NVIDIA GPUs the speedup picks up to almost 50,000x.
