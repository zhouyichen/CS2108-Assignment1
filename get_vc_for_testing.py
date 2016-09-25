import cv2
import numpy as np
from params import *
from pyimagesearch.chi2_distance import *
from pyimagesearch.deep_learning_searcher import (calculate_vc_results, read_vc_data_file)
import csv

def get_vc_for_testing(output_file, train_data, test_data):
    train_features = read_vc_data_file(train_data)
    test_features = read_vc_data_file(test_data)

    output = open(output_file, "w")

    for test_id, test_feature in test_features:
        results = {}
        test_feature = [(k, v) for (k, v) in test_feature.items()]

        results = calculate_vc_results(test_feature, train_features)

        results = sorted([(v, k) for (k, v) in results.items()])
        results = [str(w) for f in results for w in f]
        output.write("%s,%s\n" % (test_id, ",".join(results)))

    output.close()

if __name__ == "__main__":
    get_vc_for_testing(vc_results, visual_concept_train_data, visual_concept_test_data)