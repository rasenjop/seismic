#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 19:40:39 2018

@author: Fran Lopez
"""
import ctypes
import numpy.ctypeslib as ctl
import platform
import logging
import sys
import numpy as np
import time


def initialise_library():
    try:
        if platform.system() == 'Darwin':
            library = ctypes.CDLL('correlation_cuda.dylib')
        elif platform.system() == 'Windows':
            library = ctypes.CDLL('correlation_cuda.dll')
        elif platform.system() == 'Linux':
            library = ctl.load_library('correlation_multicuda.so','/home/bsc18/bsc18266/Correlation_in_CUDA/correlation_multicuda.so')
            #library = ctypes.CDLL('correlation_c.so')
    except:
        logging.error("Library has not been properly loaded")
        sys.exit(1)
    return library

def compute_correlation(events, shift, nGPUS, threshold):
    start = time.time()
    library = initialise_library()

    n_events = events.shape[0]
    # n_events = 7

    max = 0
    for i in range(n_events):
        # This section is for data coming from numpy arrays
        # if(events[0].shape[0] > max):
        #    max = events[0].shape[0]

        # This section is used for data coming from the obspy format
        if(events[0].data.shape[0] > max):
            max = events[0].data.shape[0]

    # Closer power-of-2 number of elements for the signals
    max_pad = (1 << max.bit_length())

    #max_pad = max
    # Creates the pair of arrays where the signals will be stored
    padded_events = np.ascontiguousarray(np.zeros((n_events,max_pad),dtype=np.float32),
                                         dtype=np.float32)
    padded_reversed_events = np.ascontiguousarray(np.zeros((n_events,max_pad),dtype=np.float32),
                                         dtype=np.float32)

    #So far we have two all-zero matrices that has to be filled with the signals
    for i in range(n_events):
        padded_events[i,:] = np.pad(events[i].data, (0,max_pad-events[i].data.shape[0]),'constant').astype(float)
        padded_reversed_events[i,:] = np.pad(np.flip(events[i].data,0),
                                        (0,max_pad-events[i].data.shape[0]), 'constant').astype(float)
        #padded_events[i,:] = np.pad(events[i], (0,max_pad-events[i].shape[0]),'constant')
        #padded_reversed_events[i,:] = np.pad(np.flip(events[i], 0),
        #                                (0,max_pad-events[i].shape[0]), 'constant')


    n_elements = int(n_events * (n_events+1) / 2)
    xcm_pos = np.ascontiguousarray(np.zeros(n_elements, dtype=np.float32), dtype=np.float32)
    xclags_pos = np.ascontiguousarray(np.zeros(n_elements, dtype=np.int32), dtype=np.int32)
    xcm_neg = np.ascontiguousarray(np.zeros(n_elements, dtype=np.float32), dtype=np.float32)
    xclags_neg = np.ascontiguousarray(np.zeros(n_elements, dtype=np.int32), dtype=np.int32)


    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    library.correlationCUDA.restype = None
    library.correlationCUDA.argtypes = [c_float_p, c_float_p, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_float, ctypes.c_int,
                                        c_float_p, c_int_p, c_float_p, c_int_p]

    diff_time = time.time() - start
    print("Time in correlation_lib.py before CUDA-function: " + str(diff_time))

    print("Python: About to enter the C-function")
    library.correlationCUDA(padded_events.ctypes.data_as(c_float_p),
                           padded_reversed_events.ctypes.data_as(c_float_p),
                           ctypes.c_int(n_events),
                           ctypes.c_int(max),
                           ctypes.c_int(shift),
                           ctypes.c_int(max_pad),
                           ctypes.c_float(threshold),
                           ctypes.c_int(nGPUS),
                           xcm_pos.ctypes.data_as(c_float_p),
                           xclags_pos.ctypes.data_as(c_int_p),
                           xcm_neg.ctypes.data_as(c_float_p),
                           xclags_neg.ctypes.data_as(c_int_p))

    return xcm_pos, xclags_pos, xcm_neg, xclags_neg, diff_time

def initialiseRunTime(nGPUS):
    library = initialise_library()
    library.initialiseCUDA.restype = None
    library.initialiseCUDA.argtypes = []

    library.initialiseCUDA(nGPUS)
