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

def compute_correlation(events, shift, num_threads, chunk_size):
    library = initialize_library()
    max = 0
    n_events = events.shape[0]
    #n_events = 500
    for i in range(n_events):
        #if(events[0].shape[0] > max):
        #    max = events[0].shape[0]
        if(events[0].data.shape[0] > max):
            max = events[0].data.shape[0]

    max_pad = 1 << (2*max+1).bit_length()

    padded_events = np.ascontiguousarray(np.zeros((n_events,max_pad,2),dtype=np.float32),
                                         dtype=np.float32)
    padded_reversed_events = np.ascontiguousarray(np.zeros((n_events,max_pad,2),dtype=np.float32),
                                         dtype=np.float32)

    #So far we have two all-zero matrices that has to be filled with the signals
    for i in range(n_events):
        padded_events[i,:,0] = np.pad(events[i].data, (0,max_pad-events[i].data.shape[0]),'constant').astype(float)
        padded_reversed_events[i,:,0] = np.pad(np.flip(events[i].data,0),
                                        (0,max_pad-events[i].data.shape[0]), 'constant').astype(float)
        #padded_events[i,:,0] = np.pad(events[i], (0,max_pad-events[i].shape[0]),'constant')
        #padded_reversed_events[i,:,0] = np.pad(np.flip(events[i], 0),
        #                                (0,max_pad-events[i].shape[0]), 'constant')


    xcm_pos = np.zeros((n_events, n_events), dtype=np.float32)
    xclags_pos = np.zeros((n_events, n_events), dtype=np.int32)
    xcm_neg = np.zeros((n_events, n_events), dtype=np.float32)
    xclags_neg = np.zeros((n_events, n_events), dtype=np.int32)


    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    library.correlationCPP.restype = None
    library.correlationCPP.argtypes = [c_float_p, c_float_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       c_float_p, c_int_p, c_float_p, c_int_p]

    print("Python: About to enter the C-function")
    library.correlationCPP(padded_events.ctypes.data_as(c_float_p),
                           padded_reversed_events.ctypes.data_as(c_float_p),
                           ctypes.c_int(n_events),
                           ctypes.c_int(max),
                           ctypes.c_int(shift),
                           ctypes.c_int(max_pad),
                           ctypes.c_int(num_threads),
                           ctypes.c_int(chunk_size),
                           xcm_pos.ctypes.data_as(c_float_p),
                           xclags_pos.ctypes.data_as(c_int_p),
                           xcm_neg.ctypes.data_as(c_float_p),
                           xclags_neg.ctypes.data_as(c_int_p))

    return xcm_pos, xclags_pos, xcm_neg, xclags_neg
