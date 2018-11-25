import itertools
import matplotlib.pyplot as plt
import numpy as np, os
from mpl_toolkits.mplot3d import Axes3D
import pykitti

import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import *

from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import *

from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools

from keras.utils import np_utils
from keras.optimizers import *
import pandas as pd

from keras import backend as K
K.set_image_dim_ordering('th')



class RCNN:

	def __init__(self, data, time_step, image_sequences):
		'''
		Input: image_sequences = [training sequences list, test sequences list] 
		'''
		self.time_step = time_step
		self.image_sequences = image_sequences
		self.last_image_pointer = {train: 0, test: 0}

	def initImagePaths(self):
		current_dir =os.getcwd()
		self.train_paths = [current_dir + "/" + self.path_to_images + "/sequences/" + index for index in self.image_sequences[0]]
		self.test_paths  = [current_dir + "/" + self.path_to_images + "/sequences/" + index for index in self.image_sequences[1]]

	def initDataset(self):
		loadTrainImages()
		loadTestImages()

	def loadTrainImages(self):
		# Load all the train images from all the sequences into one list 
		self.train_images_list = []
		for paths in self.train_paths:
			for image_file in os.listdir(paths):
				self.train_images_list.append[self.processRawImage(image_file)]
		if self.image_verbostiy: viewImage(self.train_images_list[-1])

	def loadTestImages(self):
		# Load all the test images from all the sequences into one list 
		self.test_images_list = []
		for paths in self.test_paths:
			for image_file in os.listdir(paths):
				self.test_images_list.append[self.processRawImage(image_file)]
		if self.image_verbostiy: viewImage(self.train_images_list[-1])

	def processRawImage(self, image_file):
		# Process the raw image
		img = cv2.imread(image_file)
        img = img/np.max(img)
        img = img - np.mean(img)
        height, width = img.shape[:2]
        img = cv2.resize(img, (width/1.5, height/1.5), fx=0, fy=0)	            

	def getBatchImages(self, train=1, batch_size=5):
		# If the batch is to be extracted from train OR test images list
		image_list = self.train_images_list if train else self.test_images_list
		stacked_image_batch = []

		# Extracting images of batch_size length using the last pointer in the list
		for index in range(self.last_image_pointer, batch_size + self.last_image_pointer):
			if ((index+1) < len(image_list)):
				stacked_image = np.concatenate([image_list[index], image_list[index+1]], -1)
				stacked_image_batch.append(stacked_image)

		# Store the last batch pointer
		if train:
			self.last_image_pointer[train] = self.last_image_pointer[train] + batch_size 
		else:
			self.last_image_pointer[test] = self.last_image_pointer[test] + batch_size

		return stacked_image_batch

	def viewImage(self, image_file):
		cv2.imread(image_file)
		cv2.imshow()
		cv2.waitkey()

	def createOptimizer(self, opt):

		lr = 0.1
		dcy = lr/self.num_epochs
		if opt=="adam":
			designed_optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=dcy)
			designed_optimizer = Adam()
		elif opt=="sgd":
			designed_optimizer = SGD(lr=0.1)

		return designed_optimizer

	def defineCnnModel(self, size_axis_1, size_axis_2, size_axis_color):

		# CNN Model 
		cnn_model = Sequential()

		cnn_model.add(Conv2D(64, (3,3), activation="relu", input_shape=(1, 28, 28), padding="valid"))
		cnn_model.add(MaxPooling2D(pool_size=(2,2)))
		cnn_model.add(Conv2D(128, (3,3), activation="sigmoid", padding="valid"))
		cnn_model.add(MaxPooling2D(pool_size=(2,2)))
		cnn_model.add(Conv2D(256, (3,3), activation="relu", padding="valid"))
		cnn_model.add(MaxPooling2D(pool_size=(2,2)))
		cnn_model.add(Conv2D(512, (3,3), activation="sigmoid", padding="valid"))
		cnn_model.add(MaxPooling2D(pool_size=(2,2)))
		cnn_model.add(Conv2D(256, (3,3), activation="relu", padding="valid"))
		cnn_model.add(MaxPooling2D(pool_size=(2,2)))
		cnn_model.add(Conv2D(128, (3,3), activation="sigmoid", padding="valid"))

		# 
		print (cnn_model.summary())
		return cnn_model		

