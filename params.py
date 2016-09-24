from pyimagesearch.colordescriptor import ColorDescriptor
import cv2 as cv2

path_to_training_data = "dataset/train/data/"
path_to_testing_data = "dataset/test/data/"

# initialize the color descriptor
cd = ColorDescriptor((18, 4, 4), (4, 4, 4))

color_hist_train_data = "colorHist_train.csv"
color_hist_test_data = "colorHist_test.csv"

deep_learning_train_data = "deepLearning_train.csv"
deep_learning_test_data = "deepLearning_test.csv"

visual_concept_train_data = "visualConcept_train.csv"
visual_concept_test_data = "visualConcept_test.csv"

sift_train_data = "siftDescriptors_train.csv"
sift_test_data = "siftDescriptors_test.csv"

bow_train_data = "bow250.pkl"

vw_results = "vw_predictions250.csv"
ch_results = "ch_predictions.csv"
dl_results = "dl_predictions.csv"
vc_results = "vc_predictions.csv"