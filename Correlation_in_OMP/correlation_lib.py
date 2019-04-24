#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 19:40:39 2018

@author: fran
"""
import ctypes
import numpy.ctypeslib as ctl
import platform
import logging
import sys
import numpy as np
import math


def initialize_library():
    try:
        if platform.system() == 'Darwin':
            library = ctypes.CDLL('correlation_c.dylib')
        elif platform.system() == 'Windows':
            library = ctypes.CDLL('correlation_c.dll')
        elif platform.system() == 'Linux':
            library = ctl.load_library('correlation_c.so','/home/bsc18/bsc18266/Correlation_in_OMPf/correlation_c.so')
            #library = ctypes.CDLL('correlation_c.so')
    except:
        logging.error("Something bad is happening..")
        sys.exit(1)
    return library

def compute_correlation(events, shift, num_threads, chunk_size, threshold, n_classes_hist):
    library = initialize_library()

    # Creates the histogram array where the values for every "class" will be stored
    max_hist = np.ascontiguousarray(np.zeros(n_classes_hist, dtype=np.int32), dtype=np.int32)
    min_hist = np.ascontiguousarray(np.zeros(n_classes_hist, dtype=np.int32), dtype=np.int32)
    # print("The delta_hist used is " + str(delta_hist) + " and the number of classes is " + str(n_classes_hist))


    n_events = events.shape[0]
    #n_events = 7 # Toy example used for debugging

    max = 0
    for i in range(n_events):
        # This section is for data coming from numpy arrays
        #if(events[0].shape[0] > max):
        #    max = events[0].shape[0]

        # This section is used for data coming from the obspy format
        if(events[0].data.shape[0] > max):
            max = events[0].data.shape[0]

    # Closer power-of-2 number of elements for the signals
    max_pad = (1 << max.bit_length())

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


    # xcm_pos = np.zeros((n_events, n_events), dtype=np.float32)
    # xclags_pos = np.zeros((n_events, n_events), dtype=np.int32)
    # xcm_neg = np.zeros((n_events, n_events), dtype=np.float32)
    # xclags_neg = np.zeros((n_events, n_events), dtype=np.int32)

    # Computes the number of elements obtained as a result - i.e. the upper matrices, including the diagonal
    n_elements = int(n_events * (n_events+1) / 2)
    xcm_pos = np.ascontiguousarray(np.zeros(n_elements, dtype=np.float32), dtype=np.float32)
    xclags_pos = np.ascontiguousarray(np.zeros(n_elements, dtype=np.int32), dtype=np.int32)
    xcm_neg = np.ascontiguousarray(np.zeros(n_elements, dtype=np.float32), dtype=np.float32)
    xclags_neg = np.ascontiguousarray(np.zeros(n_elements, dtype=np.int32), dtype=np.int32)


    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    library.correlationCPP.restype = None
    library.correlationCPP.argtypes = [c_float_p, c_float_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_int, c_float_p,
                                       c_int_p, c_float_p, c_int_p, c_int_p, c_int_p]

    print("Python: About to enter the C-function\n")
    library.correlationCPP(padded_events.ctypes.data_as(c_float_p),
                           padded_reversed_events.ctypes.data_as(c_float_p),
                           ctypes.c_int(n_events),
                           ctypes.c_int(max),
                           ctypes.c_int(shift),
                           ctypes.c_int(max_pad),
                           ctypes.c_int(num_threads),
                           ctypes.c_int(chunk_size),
                           ctypes.c_float(threshold),
                           ctypes.c_int(n_classes_hist),
                           xcm_pos.ctypes.data_as(c_float_p),
                           xclags_pos.ctypes.data_as(c_int_p),
                           xcm_neg.ctypes.data_as(c_float_p),
                           xclags_neg.ctypes.data_as(c_int_p),
                           max_hist.ctypes.data_as(c_int_p),
                           min_hist.ctypes.data_as(c_int_p))


    return xcm_pos, xclags_pos, xcm_neg, xclags_neg, max_hist, min_hist
