# import the necessary packages
import numpy as np
import cv2

class ColorDescriptor:
	def __init__(self, hsv_bins, rgb_bins):
		# store the number of hsv_bins for the 3D hsv_histogram
		self.hsv_bins = hsv_bins
		self.rgb_bins = rgb_bins

	def describe(self, image):
		# convert the hsv_image to the HSV color space and initialize
		# the features used to quantify the hsv_image
		rgb_image = image
		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []

		# grab the dimensions and compute the center of the hsv_image
		(h, w) = hsv_image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))

		# divide the hsv_image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]

		# construct an elliptical mask representing the center of the
		# hsv_image
		(axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
		ellipMask = np.zeros(hsv_image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the hsv_image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(hsv_image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)

			# extract a color hsv_histogram from the hsv_image, then update the
			# feature vector
			hsv_hist = self.hsv_histogram(hsv_image, cornerMask)
			features.extend(hsv_hist)

			rgb_hist = self.rgb_histogram(rgb_image, cornerMask)
			features.extend(rgb_hist)

		# extract a color hsv_histogram from the elliptical region and
		# update the feature vector
		hist = self.hsv_histogram(hsv_image, ellipMask)
		features.extend(hist)

		rgb_hist = self.rgb_histogram(rgb_image, ellipMask)
		features.extend(rgb_hist)

		# return the feature vector
		return np.array(features)

	def hsv_histogram(self, hsv_image, mask):
		# extract a 3D color hsv_histogram from the masked region of the
		# hsv_image, using the supplied number of hsv_bins per channel; then
		# normalize the hsv_histogram
		hist = cv2.calcHist([hsv_image], [0, 1, 2], mask, self.hsv_bins,
			[0, 180, 0, 256, 0, 256])
		hist = cv2.normalize(hist).flatten()

		# return the hsv_histogram
		return hist

	def rgb_histogram(self, image, mask):
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.rgb_bins,
			[0, 256, 0, 256, 0, 256])
		hist = cv2.normalize(hist).flatten()
		return hist