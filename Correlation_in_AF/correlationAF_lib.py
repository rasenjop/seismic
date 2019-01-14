#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 19:40:39 2018

@author: fran
"""
import ctypes
import platform
import logging
import sys
import numpy as np

#def initialize_library():
#    try:
#        if platform.system() == 'Darwin':
#            library = ctypes.CDLL('libcorrelation_AF.dylib')
#        elif platform.system() == 'Windows':
#            library = ctypes.CDLL('libcorrelation_AF.dll')
#        elif platform.system() == 'Linux':
#            library = ctypes.CDLL('libcorrelation_AF.so')
#    except:
#        logging.error("Something bad is happening..")
#        sys.exit(1)
#    return library

def initialize_library():
    try:
        if platform.system() == 'Darwin':
            library = ctypes.CDLL('./build/libcorrelation_af.dylib')
        elif platform.system() == 'Windows':
            library = ctypes.CDLL('./build/libkmeans_arrayfire.dll')
        elif platform.system() == 'Linux':
            library = ctypes.CDLL('./build/libkmeans_arrayfire.so')
    except:
        logging.error(
            "va a ser que no figura")
        sys.exit(1)
    return library



def compute_correlation(events, shift):
    library = initialize_library()
    max = 0
    n_events = events.shape[0]
    #n_events = 4
    for i in range(n_events):
        if(events[0].shape[0] > max):
            max = events[0].shape[0]
        #if(events[0].data.shape[0] > max):
        #    max = events[0].data.shape[0]
    
    #max_pad = 1 << (2*max+1).bit_length()
    
    padded_events = np.ascontiguousarray(np.zeros((n_events,max,2),dtype=np.float64),
                                         dtype=np.float64)
    #padded_reversed_events = np.ascontiguousarray(np.zeros((n_events,max_pad,2),dtype=np.float64),
    #                                     dtype=np.float64)
    
    #So far we have two all-zero matrices that has to be filled with the signals
    for i in range(n_events):
    #    padded_events[i,:,0] = np.pad(events[i].data, (0,max_pad-events[i].data.shape[0]),'constant')
    #    padded_reversed_events[i,:,0] = np.pad(np.flip(events[i].data), 
    #                                    (0,max_pad-events[i].data.shape[0]), 'constant')
        padded_events[i,:] = np.pad(events[i], (0,max-events[i].shape[0]),'constant')
        #padded_reversed_events[i,:] = np.pad(np.flip(events[i]), 
        #                                (0,max_pad-events[i].shape[0]), 'constant')
    
        
    xcm_pos = np.zeros((n_events, n_events), dtype=np.float64)
    xclags_pos = np.zeros((n_events, n_events), dtype=np.int32)
    xcm_neg = np.zeros((n_events, n_events), dtype=np.float64)
    xclags_neg = np.zeros((n_events, n_events), dtype=np.int32)
    
    
    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    library.correlationAF.restype = None
    library.correlationAF.argtypes = [c_double_p, ctypes.c_int, 
                                       c_double_p, c_int_p, 
                                       c_double_p, c_int_p]
    
    print("Python: About to enter the C-function")
    library.correlationAF(padded_events.ctypes.data_as(c_double_p),
                           ctypes.c_int(shift),
                           xcm_pos.ctypes.data_as(c_double_p),
                           xclags_pos.ctypes.data_as(c_int_p),
                           xcm_neg.ctypes.data_as(c_double_p),
                           xclags_neg.ctypes.data_as(c_int_p))
    
    return xcm_pos, xclags_pos, xcm_neg, xclags_neg