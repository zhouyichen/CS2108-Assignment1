from pyimagesearch.colordescriptor import ColorDescriptor
import cv2 as cv2

path_to_training_data = "dataset/train/data/"
path_length = len(path_to_training_data)

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))
color_hist_data = "colorHist.csv"

deep_learning_data = "deepLearning.csv"