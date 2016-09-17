# import the necessary packages
import numpy as np
import csv
import cv2

def extract_sift(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT(nfeatures = 50)
	kp, des = sift.detectAndCompute(gray, None)
	return des[:50]

class SIFT_Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	def search(self, query_image, limit = 10):
		# initialize our dictionary of results
		results = {}

		# extract features of the query
		query_features = extract_sift(query_image)

		# BFMatcher with default params
		bf = cv2.BFMatcher()

		# open the sift feature file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
				features = np.array(row[1:], dtype = np.dtype(np.float32))
				sift_features = features.reshape([-1, 128])

				matches = bf.match(sift_features, query_features)

				tresshold = 300

				# dist = [m.distance for m in matches]
				count = sum(1 for x in matches if x.distance < tresshold)
				dist = sum(x.distance for x in matches) / len(matches)
				score = dist + (50 - count) ** 0.5

				results[row[0]] = score
				
			# close the reader
			f.close()

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])
		for i in results[:10]:
			print i 
		# return our (limited) results
		return results[:limit]


# for testing purpose
if __name__ == "__main__":
	query_image = cv2.imread("../dataset/train/data/tree/0281_294263216.jpg")
	sift_searcher = SIFT_Searcher("../siftDescriptors_train.csv")
	sift_searcher.search(query_image)
