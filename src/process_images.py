
import numpy as np
import os
# from keras.models import Sequential
# from keras.layers.core import Dense
# from keras.datasets import mnist

# from keras.layers.convolutional import *
# from keras.layers.normalization import BatchNormalization
# from keras.layers import Flatten, Dropout
# from keras.preprocessing.image import ImageDataGenerator
# from keras.metrics import *

# from keras import backend as K

# from sklearn.metrics import confusion_matrix
# import itertools
# import matplotlib.pyplot as plt

# from keras.utils import np_utils
# from keras.optimizers import *
# import pandas as pd

# from keras import backend as K
# K.set_image_dim_ordering('th')

class ProcessImages():

	def __init__(self, camera_side, path_to_images):
		self.camera_side = camera_side
		self.path_to_images = path_to_images

	def completeImagesPath(self, sequence_id):
		folder_name = "image_0" if self.camera_side == "left" else "image_1"
		self.path_to_images = os.getcwd() + "/" + self.path_to_images + "/" + str(0) + str(sequence_id) + "/" + folder_name

		if not os.path.isdir(self.path_to_images):
			print ('\033[93m' +  "Image Sequence Folder does not exist: %s" %str(self.path_to_images))
			return True
		else:
			# print ('\033[92m' +  "Image Sequence Folder exists: %s" %str(self.path_to_images))
			return False

	def loadImages(self):
		
		for images in 



if __name__ == '__main__':
	camera_side = "left"
	path_to_images = "../dataset_images/sequences"
	image_object = ProcessImages(camera_side, path_to_images)
	image_object.completeImagesPath(1)