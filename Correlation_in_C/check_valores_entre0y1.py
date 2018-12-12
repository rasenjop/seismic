#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:50:11 2018

@author: fran
"""

import numpy as np
from obspy.signal.cross_correlation import xcorr, xcorr_max, correlate


s1_t = np.linspace(-0.9, 0.0, num=10)
s2_t = np.linspace(-0.3, 0.6, num=10)

xcorrij = correlate(s1_t, s2_t, 10, demean= False)
print(xcorrij)
print("SE IMPRIME LA SEGUNDA CORRELACION")
xcorrij = xcorr(s1_t, s2_t, 4, full_xcorr=True)
print(xcorrij[2])
