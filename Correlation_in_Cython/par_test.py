
from obspy.signal.cross_correlation import xcorr, xcorr_max, correlate

from numpy import (load, zeros, savetxt)

import numpy as np

import time

from par_correlate_cython_wrapper import XCM2_correlate_cython

events = load('../events_eruption_v8.npy',fix_imports=True, encoding='latin1')

tiempos = np.zeros(1)

for i in range(1):
	start = time.time()

	xcm_pos, xclags_pos, xcm_neg, xclags_neg = XCM2_correlate_cython(events)

	diff_time = time.time() - start
	print("Execution finished in: " + str(diff_time))
	tiempos[i] = diff_time
print("Max: " + str(tiempos.max()) + " Mean: "+ str(tiempos.mean()) + " Min: "+str(tiempos.min()))

savetxt('xcm_v1_neg.txt',xcm_neg, delimiter='\t', fmt='%6.3f')

savetxt('xcl_v1_neg.txt',xclags_neg, delimiter='\t', fmt='%6.0f')

savetxt('xcm_v1_pos.txt',xcm_pos, delimiter='\t', fmt='%6.3f')

savetxt('xcl_v1_pos.txt',xclags_pos, delimiter='\t', fmt='%6.0f')
