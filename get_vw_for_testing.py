import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from params import *
import glob
from pyimagesearch.bow_searcher import search_for_match
import csv
import sys

def get_vw_for_testing(output_file, sift_data, sift_bow_data):
    # Load the classifier, class names, scaler, number of clusters and vocabulary 
    image_ids, im_features, k, voc = joblib.load(sift_bow_data)

    # Get all the path to the images and save them in a list
    # imageID and the corresponding label in imageID

    image_classes = []
    class_id = 0

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
            
            des_list.append((row[0], sift_features))

            image_classes += [class_id] * 30
            class_id += 1

        # close the reader
        f.close()

    total_number_of_images = len(des_list)

    # Calculate the histogram of features
    print "Calculate the histogram of features"
    test_features = np.zeros((total_number_of_images, k), "float32")
    for i in xrange(total_number_of_images):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            test_features[i, w] += 1
        test_features[i, :] = cv2.normalize(test_features[i, :]).T

    output = open(output_file, "w")
    for i in range(len(test_features)):
        results = search_for_match(image_ids, im_features, test_features[i])
        results = sorted([(v, k) for (k, v) in results.items()])
        results = [str(w) for f in results for w in f]
        output.write("%s,%s\n" % (des_list[i][0], ",".join(results)))        

    output.close()

if __name__ == "__main__":
    get_vw_for_testing(vw_results, sift_test_data, bow_train_data)


