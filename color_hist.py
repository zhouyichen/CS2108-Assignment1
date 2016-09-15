# USAGE
# python index.py --dataset dataset --index index.csv

# import the necessary packages
from params import *
import argparse
import glob

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = False, default='dataset',
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = False, default=color_hist_data,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())

# open the output index file for writing
output = open(args["index"], "w")

path_to_training_data = "dataset/train/data/"
path_length = len(path_to_training_data)

# use glob to grab the image paths and loop over them
for categoryPath in glob.glob(path_to_training_data + "*"):
	category = categoryPath[path_length:]
	for imagePath in glob.glob(categoryPath + "/*.jpg"):
		
		# extract the image ID (i.e. the unique filename) from the image
		# path and load the image itself
		imageID = category + "/" + imagePath[imagePath.rfind("/") + 1:]
		image = cv2.imread(imagePath)

		# describe the image
		features = cd.describe(image)

		# write the features to file
		features = [str(f) for f in features]
		output.write("%s,%s\n" % (imageID, ",".join(features)))
		

# close the index file
output.close()