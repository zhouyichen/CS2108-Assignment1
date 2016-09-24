import cv2
import numpy as np
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from params import *
import glob
import csv
import sys

def bow_on_data(output_file, cluster_data):

	image_ids, des_list, k, voc = joblib.load(cluster_data)

	total_number_of_images = len(image_ids)

	# Calculate the histogram of features
	print "Calculate the histogram of features"
	im_features = np.zeros((total_number_of_images, k), "float32")
	for i in xrange(total_number_of_images):
	    words, distance = vq(des_list[i], voc)
	    for w in words:
	        im_features[i, w] += 1
	    # im_features[i, :] = cv2.normalize(im_features[i, :]).T

	joblib.dump((image_ids, im_features, k, voc), output_file, compress=3)

if __name__ == "__main__":
	k = int(sys.argv[1])
	bow_on_data("bow" + str(k) + ".pkl", "sift_cluster_" + str(k) + ".pkl")
