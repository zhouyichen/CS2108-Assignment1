# import the necessary packages
import numpy as np
import csv

class ColorHistSearcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath
		self.features_list = []

		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)

			# loop over the rows in the index
			for row in reader:
				self.features_list.append(row)
				
			# close the reader
			f.close()

	def search(self, queryFeatures, weight = 1):
		# initialize our dictionary of results
		results = {}

			# loop over the rows in the index
		for row in self.features_list:

			features = np.array(row[1:], dtype = np.dtype(float))
			d = self.chi2_distance(features, queryFeatures)

			# now that we have the distance between the two feature
			# vectors, we can udpate the results dictionary -- the
			# key is the current image ID in the index and the
			# value is the distance we just computed, representing
			# how 'similar' the image in the index is to our query
			results[row[0]] = d * weight

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		# results = sorted([(v, k) for (k, v) in results.items()])

		# return our (limited) results
		return results

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d =  0.5 * np.sum(np.square(histA - histB) / (histA + histB + eps))

		# return the chi-squared distance
		return d