#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:23:53 2018

@author: fran
"""

import numpy as np

import time

from correlationAF_lib import compute_correlation_AF

#events = np.load('events_eruption_v8.npy',fix_imports=True, encoding='latin1')

#events = np.load('events_2017-18_VCH1.npy', fix_imports=True, encoding='latin1')

# events=np.random.rand(10, 15)
events = np.ones((10,15), dtype=np.float64)
print(events)


tiempos = np.zeros(1)

for i in range(1):
	start = time.time()

	xcm_pos, xclags_pos, xcm_neg, xclags_neg = compute_correlation_AF(events, 3)

	diff_time = time.time() - start
	print("Execution finished in: " + str(diff_time))
	tiempos[i] = diff_time
print("Max: " + str(tiempos.max()) + " Mean: "+ str(tiempos.mean()) + " Min: "+str(tiempos.min()))

# np.savetxt('xcm_v1_neg.txt',xcm_neg, delimiter='\t', fmt='%6.3f')
#
# np.savetxt('xcl_v1_neg.txt',xclags_neg, delimiter='\t', fmt='%6.0f')
#
# np.savetxt('xcm_v1_pos.txt',xcm_pos, delimiter='\t', fmt='%6.3f')
#
# np.savetxt('xcl_v1_pos.txt',xclags_pos, delimiter='\t', fmt='%6.0f')
