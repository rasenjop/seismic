#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:15:55 2018

@author: fran
"""

import numpy as np
from obspy.signal.cross_correlation import xcorr, xcorr_max, correlate

signal1_t = np.linspace(0.0, 9, num=10)
signal2_t = np.linspace(1.0, 19.0, num=10)
signal3_t = np.linspace(30.0, 49.0, num=10)
signal4_t = np.linspace(51.0, .0, num=10)

signal1_b = np.hstack([signal1_t,np.zeros(11)])
signal2_b = np.hstack([signal2_t,np.zeros(11)])
signal3_b = np.hstack([signal3_t,np.zeros(11)])
signal4_b = np.hstack([signal4_t,np.zeros(11)])

events = np.vstack([signal1_b, signal2_b, signal3_b, signal4_b])

n_events = events.shape[0]
    
xcorr_vals_pos = np.zeros((n_events,n_events))

xcorr_lags_pos = np.zeros((n_events,n_events))

xcorr_vals_neg = np.zeros((n_events,n_events))

xcorr_lags_neg = np.zeros((n_events,n_events))

shift = 5

for i in range(n_events):

    #print (str(i) + ' of ' + str(n_events))

    for j in range(i, n_events):

        xcorrij = xcorr(events[i], events[j], shift, full_xcorr=True)
        #print(xcorrij[2].shape)
        #Return absolute max XC, including negative values

        if xcorrij[1]<0.:

            #Return highest positive XC

            xcorr_lags_pos[i,j], xcorr_vals_pos[i,j] = xcorr_max(xcorrij[2], abs_max=False)
            xcorr_lags_neg[i,j], xcorr_vals_neg[i,j] = xcorr_max(xcorrij[2], abs_max=True)

        else:

            xcorr_vals_neg[i,j] = xcorrij[2].min()
            xcorr_lags_neg[i,j] = np.where(xcorrij[2]== xcorrij[2].min())[0][0]-shift
            xcorr_lags_pos[i,j], xcorr_vals_pos[i,j] = xcorr_max(xcorrij[2], abs_max=False)



#xcorrij = xcorr(events[i], events[j], 250, full_xcorr=True)
#xcorrij = correlate(signal1_t, signal2_t, 10, domain='time', demean=False)
#xcorrij_xcorr = xcorr(signal1_b, signal2_b, 10, full_xcorr = True)
#print(xcorrij.shape)
#
#print(signal1_t)
#print(signal2_t)
#print(xcorrij)
#
#print(xcorrij_xcorr)
