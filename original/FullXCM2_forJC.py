# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 13:57:21 2015

@author: abell5
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 11:42:28 2015

@author: abell5
"""
from obspy.signal.cross_correlation import xcorr, xcorr_max
from numpy import (load, zeros, savetxt)

events = load('events_eruption_v8.npy',fix_imports=True,encoding='latin1')

########################
def XCM2(events):
    n_events = len(events)
    #print n_events

    xcorr_vals_pos = zeros((n_events,n_events))
    xcorr_lags_pos = zeros((n_events,n_events))
    xcorr_vals_neg = zeros((n_events,n_events))
    xcorr_lags_neg = zeros((n_events,n_events))

    for i in range(n_events):
        print ("{} of {}".format(i,n_events))
        for j in range(i, n_events):
            xcorrij = xcorr(events[i], events[j], 250, full_xcorr=True)
            #Return absolute max XC, including negative values
            xcorr_lags_neg[i,j] = xcorrij[0]
            xcorr_vals_neg[i,j] = xcorrij[1]
            if xcorrij[1]<0.:
                #Return highest positive XC
                xcorr_lags_pos[i,j], xcorr_vals_pos[i,j] = xcorr_max(xcorrij[2], abs_max=False)
            else:
                xcorr_lags_pos[i,j] = xcorrij[0]
                xcorr_vals_pos[i,j] = xcorrij[1]

    return xcorr_vals_pos, xcorr_lags_pos, xcorr_vals_neg, xcorr_lags_neg

xcm_pos, xclags_pos, xcm_neg, xclags_neg = XCM2(events)

savetxt('xcm_v1_neg.txt',xcm_neg, delimiter='\t', fmt='%6.3f')
savetxt('xcl_v1_neg.txt',xclags_neg, delimiter='\t', fmt='%6.0f')
savetxt('xcm_v1_pos.txt',xcm_pos, delimiter='\t', fmt='%6.3f')
savetxt('xcl_v1_pos.txt',xclags_pos, delimiter='\t', fmt='%6.0f')
