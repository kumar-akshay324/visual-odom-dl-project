import itertools
import matplotlib.pyplot as plt
import numpy as np, os, sys
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import cv2
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

class RCNN:

	def __init__(self, cnn_model_params ,rnn_model_params, data_params, operation_flags):
		'''
		Input: 	data_params 		= 	[path_to_dataset, image_sequences, image_ratio,, image_verbosity]
										image_sequences = [training sequences list, test sequences list]
				rnn_model_params	= 	[time_step,	LSTM_nodes, .....]
				cnn_model_params	= 	[batch_size, ]
				operation_flags		= 	[debug_verbosity]
		'''
		self.batch_size 		= cnn_model_params[0]
		self.time_step 			= rnn_model_params[0]
		self.path_to_dataset 	= data_params[0] 
		self.image_sequences 	= data_params[1]
		self.img_ratio 			= data_params[2]
		self.image_verbosity 	= data_params[-1]
		self.last_image_pointer = {"train": 0, "test": 0}
		self.debug_verbosity 	= operation_flags[0]

	# ------------- Data Processing -------------------- #

	def initImagePaths(self):
		current_dir =os.getcwd()
		partial_path = current_dir + "/" + self.path_to_dataset + "/sequences/"
		self.train_paths = [ partial_path + str(index) + "/image_0" for index in self.image_sequences[0]]
		self.test_paths  = [ partial_path + str(index) + "/image_0" for index in self.image_sequences[1]]

		for paths in self.train_paths:
			if not os.path.exists(paths):
				print ("Train Path doesn't exist: %s" %paths)
				sys.exit()
		for paths in self.test_paths:
			if not os.path.exists(paths):
				print ("Test Path doesn't exist: %s" %paths)
				sys.exit()
		if self.debug_verbosity:
			print ("Got image paths correctly")

	def initDataset(self):
		self.initTrainImages()
		self.initTestImages()

	def initTrainImages(self):
		self.train_path_image_lengths = []
		for paths in self.train_paths:
			self.train_path_image_lengths.append(len(os.listdir(paths)))
		if self.debug_verbosity:
			print ("Number of images in the train paths %s" %str(self.train_path_image_lengths)) 

	def initTestImages(self):
		self.test_path_image_lengths = []
		for paths in self.test_paths:
			self.test_path_image_lengths.append(len(os.listdir(paths)))
		if self.debug_verbosity:
			print ("Number of images in the test paths %s" %str(self.test_path_image_lengths)) 
		
	def getTrainImage(self, image_pointer):
		# Load all the train images from all the sequences into one list 
		for path_index, paths in enumerate(self.train_path_image_lengths):
			images_length_in_path = sum(self.train_path_image_lengths[0:(path_index+1)])
			for image_index_in_path, image_file in enumerate(os.listdir(paths)):
				image_index_overall = images_length_in_path + image_index_in_path
				if (image_pointer == image_index_overall):
					img = self.processRawImage(paths + "/" + image_file)
		if self.image_verbosity:
			print ("Successfully extracted training images")
			viewImage(img)
		return img

	def getTestImage(self, image_pointer):
		# Load all the test images from all the sequences into one list 
		for path_index, paths in enumerate(self.test_path_image_lengths):
			images_length_in_path = sum(self.test_path_image_lengths[0:(path_index+1)])
			for image_index_in_path, image_file in enumerate(os.listdir(paths)):
				image_index_overall = images_length_in_path + image_index_in_path
				if (image_pointer == image_index_overall):
					img = self.processRawImage(paths + "/" + image_file)
		if self.image_verbosity:
			print ("Successfully extracted testing images")
			viewImage(img)
		return img

	def processRawImage(self, image_file):
		# Process the raw image
		img = cv2.imread(image_file)
		img = img / np.max(img)
		img = img - np.mean(img)
		height, width = img.shape[:2]
		img = cv2.resize(img, (int(width/self.img_ratio), int(height/self.img_ratio)), fx=0, fy=0)
		return img

	def getBatchImages(self, train=1):
		# If the batch is to be extracted from train OR test images list
		image_list = self.train_images_list if train else self.test_images_list
		stacked_image_batch = []

		# Extracting images of batch_size length using the last pointer in the list
		for index in range(self.last_image_pointer, self.batch_size + self.last_image_pointer, self.time_step):
			if ((index+1) < len(image_list)):
				stacked_image = np.concatenate([getTrainImage(index), getTrainImage(index+1)], -1)
				stacked_image_batch.append(stacked_image)

		# Store the last batch pointer
		if train:
			self.last_image_pointer["train"] = self.last_image_pointer["train"] + self.batch_size 
		else:
			self.last_image_pointer["test"] = self.last_image_pointer["test"] + self.batch_size

		return stacked_image_batch

	def viewImage(self, image_file):
		cv2.imread(image_file)
		cv2.imshow()
		cv2.waitkey()

	# ------------- Deep Learning Model -------------------- #

	def createOptimizer(self, opt):
		'''Create an optimizer from the available ones in Keras'''
		lr = 0.001
		dcy = lr/self.num_epochs
		if opt=="adam":
			designed_optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=dcy)
			designed_optimizer = Adam()
		elif opt=="sgd":
			designed_optimizer = SGD(lr=0.1)

		return designed_optimizer

	def defineCnnModel(self):
		'''Basic Convolutional Neural Network for processing stacked images'''
		# CNN Model 
		cnn_model = Sequential()

		# Trapezoidal Shaped Convolutional Neural Network
		cnn_model.add(TimeDistributed(Conv2D(64, (3,3), activation="relu", input_shape=(self.width, self.height, 2), padding="valid", strides=(2,2))))
		cnn_model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

		cnn_model.add(TimeDistributed(Conv2D(128, (3,3), activation="sigmoid", padding="valid", strides=(2,2))))
		cnn_model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

		cnn_model.add(TimeDistributed(Conv2D(256, (3,3), activation="relu", padding="valid", strides=(1,1))))
		cnn_model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

		cnn_model.add(TimeDistributed(Conv2D(256, (3,3), activation="relu", padding="valid", strides=(1,1))))
		cnn_model.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))

		cnn_model.add(TimeDistributed(Conv2D(128, (3,3), activation="sigmoid", padding="valid")))

		cnn_model.add(TimeDistributed(Flatten()))

		print ("CNN Model")
		print (cnn_model.summary())		
		self.cnn_model = cnn_model

	def defineLSTM(self):
		'''Additional LSTM network after CNN layer'''
		lstm_model = Sequential()
		lstm_model.add(LSTM(32, input_shape=(5,6), activation='tanh', recurrent_activation='hard_sigmoid',
		implementation=1, return_sequences=False, return_state=False, stateful=True, unroll=False))
		lstm_model.add(LSTM(8))
		self.lstm_model = lstm_model

	def defineFinalModel(self):

		self.rcnn_model = self.cnn_model.add(self.lstm_model)
		# self.rcnn_model.add(Dense(6, kernel_initializer=’normal’, activation=’linear’))

	# def compileModel(self):
	# 	self.rcnn_model.compile(loss=’mse’,optimizer =self.createOptimizer("adam"),metrics=[‘accuracy’])

	# def trainModel(self):
	# 	self.rcnn_model.fit(X,y,epochs=1,batch_size=self.batch_size,validation_split=0.05,verbose=1);

	# def obtainResult(self):
	# 	scores = model.evaluate(X,y,verbose=1,batch_size=5)
	# 	print(‘Accurracy: {}’.format(scores[1])) 

	def execute(self):
		self.initImagePaths()
		self.initDataset()


if __name__ == '__main__':
	cnn_model_params = [5]
	rnn_model_params = [3]
	path_to_poses = "../dataset_images"	
	image_sequences = [["00"], ["11"]]
	img_ratio, image_verbosity, debug_verbosity = 1, 1, 1
	operation_flags = [debug_verbosity]
	data_params = [path_to_poses, image_sequences, img_ratio, image_verbosity]
	rcnn_object = RCNN(cnn_model_params, rnn_model_params, data_params, operation_flags)
	rcnn_object.execute()