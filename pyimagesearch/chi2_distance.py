import numpy as np

def chi2_distance(histA, histB, eps = 1e-10):
	# compute the chi-squared distance
	return 0.5 * np.sum(np.square(histA - histB) / (histA + histB + eps))
