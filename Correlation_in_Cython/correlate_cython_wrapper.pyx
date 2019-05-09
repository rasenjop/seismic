"""
Created on Thu Nov 22 19:25:36 2018

@author: fran
"""

from obspy.signal.cross_correlation import xcorr, xcorr_max, correlate

from numpy import (load, zeros, savetxt)

import numpy as np
cimport numpy as np

########################

def XCM2_correlate_cython(events):

    #n_events = len(events)

    #print n_events

    cdef int n_events=1000

    xcorr_vals_pos = np.zeros((n_events,n_events))

    xcorr_lags_pos = np.zeros((n_events,n_events))

    xcorr_vals_neg = np.zeros((n_events,n_events))

    xcorr_lags_neg = np.zeros((n_events,n_events))

    cdef int i=0

    for i in range(n_events):

        #print (str(i) + ' of ' + str(n_events))

        for j in range(i, n_events):

            #xcorrij = xcorr(events[i], events[j], 250, full_xcorr=True)
            xcorrij = correlate(events[i], events[j], 250, 'fft')
            # correlate returns a numpy array formed by doubles

            #Return absolute max XC, including negative values

            xcorr_lags_neg[i,j], xcorr_vals_neg[i,j] = xcorr_max(xcorrij)


            if xcorr_vals_neg[i,j]<0.0:

                #Return highest positive XC

                xcorr_lags_pos[i,j], xcorr_vals_pos[i,j] = xcorr_max(xcorrij, abs_max=False)

            else:

                xcorr_lags_pos[i,j] = xcorr_lags_neg[i,j]

                xcorr_vals_pos[i,j] = xcorr_vals_neg[i,j]



    return xcorr_vals_pos, xcorr_lags_pos, xcorr_vals_neg, xcorr_lags_neg
