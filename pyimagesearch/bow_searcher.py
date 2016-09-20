# import the necessary packages
import numpy as np
import csv
import cv2
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import glob
import sys

def extract_sift(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT()
	kp, des = sift.detectAndCompute(gray, None)
	return des

def search_for_match(image_ids, im_features, test_features, weight):
    results = {}
    for i in range(len(im_features)):
        d = chi2_distance(im_features[i], test_features)
        results[image_ids[i]] = d * weight

    # results = sorted([(v, k) for (k, v) in results.items()])
    return results

def chi2_distance(histA, histB, eps = 1e-10):
    # compute the chi-squared distance
    return 0.5 * np.sum(np.square(histA - histB) / (histA + histB + eps))

class BOW_Searcher:
	def __init__(self, sift_bow_data):
		# store our index path
		self.sift_bow_data = sift_bow_data
		self.image_ids, self.im_features, self.k, self.voc = joblib.load(self.sift_bow_data)


	def search(self, query_image, weight = 1):
		# initialize our dictionary of results
		results = {}

		# extract features of the query
		query_features = extract_sift(query_image)

		test_features = np.zeros(self.k, "float32")

		words, distance = vq(query_features, self.voc)
		for w in words:
			test_features[w] += 1
		test_features = cv2.normalize(test_features).T

		results = search_for_match(self.image_ids, self.im_features, test_features, weight)

		return results

# for testing purpose
if __name__ == "__main__":
	k = int(sys.argv[1])
	query_image = cv2.imread("../dataset/test/data/plane/0427_1308585041.jpg")
	bow_searcher = BOW_Searcher("../bow" + str(k) + ".pkl")
	results = bow_searcher.search(query_image)
	for i in results:
		print i


