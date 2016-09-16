from params import *
import glob

def extract_sift(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT(nfeatures = 50)
	kp, des = sift.detectAndCompute(gray, None)
	return des[:50].flatten()

def sift_on_data(output_file, input_path):
	# open the output file for writing
	output = open(output_file, "w")
	path_length = len(input_path)
	# use glob to grab the image paths and loop over them
	for categoryPath in glob.glob(input_path + "*"):
		category = categoryPath[path_length:]
		for imagePath in glob.glob(categoryPath + "/*.jpg"):
			
			# extract the image ID (i.e. the unique filename) from the image
			# path and load the image itself
			imageID = category + "/" + imagePath[imagePath.rfind("/") + 1:]
			image = cv2.imread(imagePath)

			# describe the image
			features = extract_sift(image)

			# write the features to file
			features = [str(f) for f in features]
			output.write("%s,%s\n" % (imageID, ",".join(features)))
		
	# close the index file
	output.close()

if __name__ == "__main__":
	sift_on_data(sift_train_data, path_to_training_data)
	sift_on_data(sift_test_data, path_to_testing_data)

