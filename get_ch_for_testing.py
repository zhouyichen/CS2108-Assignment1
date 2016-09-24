import cv2
import numpy as np
from params import *
from pyimagesearch.chi2_distance import *
import csv

def read_file(data):
    results = []

    # open the sift feature file for reading
    with open(data) as f:
        # initialize the CSV reader
        reader = csv.reader(f)

        # loop over the rows in the index
        for row in reader:
            features = np.array(row[1:], dtype = np.dtype(np.float32))
            results.append((row[0], features))

        # close the reader
        f.close()
    return results

def get_ch_for_testing(output_file, train_data, test_data):
    train_features = read_file(train_data)
    test_features = read_file(test_data)

    output = open(output_file, "w")
    for test_img in test_features:
        results = {}
        test_id, test_feature = test_img
        for train_img in train_features:
            train_id, train_feature = train_img
            d = chi2_distance(test_feature, train_feature)
            results[train_id] = d
        results = sorted([(v, k) for (k, v) in results.items()])
        results = [str(w) for f in results for w in f]
        output.write("%s,%s\n" % (test_id, ",".join(results)))

    output.close()

if __name__ == "__main__":
    get_ch_for_testing(ch_results, color_hist_train_data, color_hist_test_data)
