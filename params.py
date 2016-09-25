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

bow_train_data = "bow300.pkl"

vw_results = "vw_predictions300.csv"
ch_results = "ch_predictions.csv"
dl_results = "dl_predictions.csv"
vc_results = "vc_predictions.csv"


CH = 'Color Histogram'
DL = 'Deep Learning'
VW = 'Visual Keyword'
VC = 'Visual Concept'


methods = [CH, VW, DL, VC]

weights_for_methods = {
	CH: 0.1,
	VW: 0.72,
	DL: 1000,
	VC: 5000
}

TOP_RESULTS_NUMBER = 16
TOTAL_NUMBER_OF_TESTS = 300
NUMBER_OF_CATEGORIES = 30
NUMBER_OF_TEST_IN_EACH_CAT = 10

category_names = [
	'alley',
	'antlers',
	'baby',
	'balloons',
	'beach',
	'bear',
	'birds',
	'boats',
	'cars',
	'cat',
	'computer',
	'coral',
	'dog',
	'fish',
	'flags',
	'flowers',
	'horses',
	'leaf',
	'plane',
	'rainbow',
	'rocks',
	'sign',
	'snow',
	'tiger',
	'tower',
	'train',
	'tree',
	'whales',
	'window',
	'zebra'
]