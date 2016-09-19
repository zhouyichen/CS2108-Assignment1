import cv2
import numpy as np
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from params import *
import glob
import csv
import sys

def bow_on_data(output_file, sift_data):

	# Get all the path to the images and save them in a list
	# imageID and the corresponding label in imageID

	image_classes = []
	class_id = 0
	image_ids = []

	# List where all the descriptors are stored
	des_list = []

	print "Extracting SIFT"

	# open the sift feature file for reading
	with open(sift_data) as f:
		# initialize the CSV reader
		reader = csv.reader(f)

		# loop over the rows in the index
		for row in reader:
			# parse out the image ID and features, then compute the
			# chi-squared distance between the features in our index
			# and our query features
			features = np.array(row[1:], dtype = np.dtype(np.float32))
			sift_features = features.reshape([-1, 128])
			
			image_ids.append(row[0])
			des_list.append(sift_features)

			image_classes += [class_id] * 30
			class_id += 1

		# close the reader
		f.close()

	# Stack all the descriptors vertically in a numpy array
	print "Stack all the descriptors vertically in a numpy array"
	# for image_path, descriptor in des_list[:]:
	descriptors = np.vstack(des_list)

	# Perform k-means clustering
	print "Perform k-means clustering: ", k
	voc, variance = kmeans(descriptors, k, 1) 

	joblib.dump((image_ids, des_list, k, voc), output_file, compress=3)

if __name__ == "__main__":
	k = int(sys.argv[1])
	bow_on_data("sift_cluster_" + str(k) + ".pkl", sift_train_data)
