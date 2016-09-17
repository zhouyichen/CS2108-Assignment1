# import the necessary packages
import numpy as np
import csv
import cv2

def extract_sift(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT(nfeatures = 50)
	kp, des = sift.detectAndCompute(gray, None)
	return des[:50]

