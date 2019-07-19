#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:23:53 2018

@author: fran
"""

import numpy as np
import sys
import time

from scipy.sparse import csr_matrix, save_npz

from correlation_lib import compute_correlation, initialiseRunTime

num_threads = int(sys.argv[1])
threshold = float(sys.argv[2])

initialiseRunTime()

events = np.load('events_eruption_v8.npy',fix_imports=True, encoding='latin1')

#events=np.ones((7, 10),dtype=np.float32)
times_loop = 1
tiempos = np.zeros(times_loop)

print("The number of threads is: " + str(num_threads))

for i in range(times_loop):
	start = time.time()
	xcm_pos, xclags_pos, xcm_neg, xclags_neg = compute_correlation(events, 256, num_threads, threshold)
	diff_time = time.time() - start
	print("Execution finished in: " + str(diff_time))
	tiempos[i] = diff_time

print("Max: " + str(tiempos.max()) + " Mean: "+ str(tiempos.mean()) + " Min: "+str(tiempos.min()))

xcm_pos = csr_matrix(xcm_pos, dtype=np.float32)
xclags_pos = csr_matrix(xclags_pos, dtype=np.int32)
xcm_neg = csr_matrix(xcm_neg, dtype=np.float32)
xclags_neg = csr_matrix(xclags_neg, dtype=np.int32)

save_npz('out_xcm_pos.npz', xcm_pos, compressed=True)
save_npz('out_xcl_pos.npz', xclags_pos, compressed=True)
save_npz('out_xcm_neg.npz', xcm_neg, compressed=True)
save_npz('out_xcl_neg.npz', xclags_neg, compressed=True)

# np.savetxt('xcm_v1_neg.txt',xcm_neg, delimiter='\t', fmt='%6.3f')
#
# np.savetxt('xcl_v1_neg.txt',xclags_neg, delimiter='\t', fmt='%6.0f')
#
# np.savetxt('xcm_v1_pos.txt',xcm_pos, delimiter='\t', fmt='%6.3f')
#
# np.savetxt('xcl_v1_pos.txt',xclags_pos, delimiter='\t', fmt='%6.0f')
