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
from correlation_lib import compute_correlation
import matplotlib.pyplot as plt


num_threads = int(sys.argv[1])
chunk_size = int(sys.argv[2])
threshold = float(sys.argv[3])
n_classes_hist = int(sys.argv[4])

if n_classes_hist < 2:
	n_classes_hist = 20 # default value
	print("n_classes_hist should not be inferior than 2. Assigned to default value (20).")

events = np.load('events_eruption_v8.npy',fix_imports=True, encoding='latin1')

#events=np.ones((7, 10),dtype=np.float32)

tiempos = np.zeros(1)

for i in range(1):
	start = time.time()
	xcm_pos, xclags_pos, xcm_neg, xclags_neg, max_hist, min_hist =\
	    compute_correlation(events, 250, num_threads, chunk_size, threshold, n_classes_hist)
	diff_time = time.time() - start
	print("Execution finished in: " + str(diff_time))
	tiempos[i] = diff_time

print("Max: " + str(tiempos.max()) + " Mean: "+ str(tiempos.mean()) + " Min: "+str(tiempos.min()))

xcm_pos = csr_matrix(xcm_pos, dtype=np.float32)
xclags_pos = csr_matrix(xclags_pos, dtype=np.int32)
xcm_neg = csr_matrix(xcm_neg, dtype=np.float32)
xclags_neg = csr_matrix(xclags_neg, dtype=np.int32)
max_hist = csr_matrix(max_hist, dtype=np.int32)
min_hist = csr_matrix(min_hist, dtype=np.int32)

save_npz('out_xcm_pos.npz', xcm_pos, compressed=True)
save_npz('out_xcl_pos.npz', xclags_pos, compressed=True)
save_npz('out_xcm_neg.npz', xcm_neg, compressed=True)
save_npz('out_xcl_neg.npz', xclags_neg, compressed=True)
save_npz('out_xmax_hist.npz', max_hist, compressed=True)
save_npz('out_xmin_hist.npz', min_hist, compressed=True)


# xcm_dense = xcm_pos.toarray()
# print(xcm_dense.shape)

#np.savez_compressed('output_matrices', xcm_pos=xcm_pos, xcl_pos=xclags_pos, xcm_neg=xcm_neg, xcl_neg=xclags_neg)

# np.savetxt('xcm_v1_neg.txt', xcm_neg, delimiter='\t', fmt='%6.3f')
#
# np.savetxt('xcl_v1_neg.txt', xclags_neg, delimiter='\t', fmt='%6.0f')
#
# np.savetxt('xcm_v1_pos.txt', xcm_pos, delimiter='\t', fmt='%6.3f')
#
# np.savetxt('xcl_v1_pos.txt', xclags_pos, delimiter='\t', fmt='%6.0f')
