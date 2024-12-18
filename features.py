# import the necessary packages
from imutils import paths
import numpy as np
import cv2

def quantify_image(image, bins=(4, 6, 3)):
	# compute a 3D color histogram over the image and normalize it
	hist = cv2.calcHist([image], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()

	# return the histogram
	return hist

def load_dataset(datasetPath, bins):
	# grab the paths to all images in our dataset directory, then
	# initialize our lists of images
	# imagePaths = "E:\\SDV\\Tutorial\\ImageAnomalyDetection-main\\forest"
	imagePaths = list(paths.list_images(datasetPath))
	data = []
	failed = []
 
	# loop over the image paths
	for imagePath in imagePaths:
		# load the image and convert it to the HSV color space
		image = cv2.imread(imagePath)
		try:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			features = quantify_image(image, bins)
			# quantify the image and update the data list
			data.append(features)
		except:
			failed.append(imagePath)

	# return our data list as a NumPy array
	return np.array(data), failed