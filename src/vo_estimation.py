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
				operation_flags		= 	[debug_verbosity, [train_image_start_pointer, test_image_start_pointer]]
		'''
		self.batch_size 		= cnn_model_params[0]
		self.time_step 			= rnn_model_params[0]
		self.path_to_dataset 	= data_params[0] 
		self.image_sequences 	= data_params[1]
		self.img_ratio 			= data_params[2] if data_params[2] else 1 
		self.image_verbosity 	= data_params[-1]
		self.last_image_pointer = {"train": train_image_start_pointer, "test": test_image_start_pointer}
		self.debug_verbosity 	= operation_flags[0]

	# ------------- Data Processing -------------------- #

	def initImagePaths(self):
		'''Look into the training and testing image paths and confirm their existence'''
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
		'''Estimate the training and testing image contents'''
		self.initTrainImages()
		self.initTestImages()

	def initTrainImages(self):
		self.train_path_image_lengths = []
		for paths in self.train_paths:
			self.train_path_image_lengths.append(len(os.listdir(paths)))
		self.total_train_images = sum(self.train_path_image_lengths)
		if self.debug_verbosity:
			print ("Number of images in the train paths %s" %str(self.train_path_image_lengths)) 

	def initTestImages(self):
		self.test_path_image_lengths = []
		for paths in self.test_paths:
			self.test_path_image_lengths.append(len(os.listdir(paths)))
		self.total_test_images = sum(self.test_path_image_lengths)
		if self.debug_verbosity:
			print ("Number of images in the test paths %s" %str(self.test_path_image_lengths)) 
		
	def getTrainImage(self, image_pointer):
		'''Return the training image for the corresponding pointer'''
		success_flag = 0
		img = np.zeros((10,10,3), np.uint8)
		for path_index, paths in enumerate(self.train_paths):
			images_length_in_path = sum(self.train_path_image_lengths[0:(path_index)])

			for image_index_in_path, image_file in enumerate(os.listdir(paths)):
				image_index_overall = images_length_in_path + image_index_in_path
				if (image_pointer == image_index_overall):
					img = self.processRawImage(paths + "/" + image_file)
					success_flag = 1
					break	
		if success_flag == 1:
			if self.debug_verbosity: print ("Successfully processed raw training image. Index: [%d, %d]" %(path_index, image_index_in_path))
		else:
			if self.debug_verbosity: print ("Could not process training image")
			sys.exit()

		if self.image_verbosity and self.debug_verbosity:
			self.viewImage(img)
		return img

	def getTestImage(self, image_pointer):
		'''Return the testing image for the corresponding pointer'''
		success_flag = 0
		img = np.zeros((10,10,3), np.uint8)
		for path_index, paths in enumerate(self.test_paths):
			images_length_in_path = sum(self.test_path_image_lengths[0:(path_index)])
			for image_index_in_path, image_file in enumerate(os.listdir(paths)):
				image_index_overall = images_length_in_path + image_index_in_path
				if (image_pointer == image_index_overall):
					img = self.processRawImage(paths + "/" + image_file)
					success_flag = 1
					break

		if success_flag==1:
			if self.debug_verbosity: print ("Successfully processed raw testing image. Index: [%d, %d]" %(path_index, image_index_in_path))
		else:
			if self.debug_verbosity: print ("Could not process testing image")
			sys.exit()
		if self.image_verbosity and self.debug_verbosity:
			self.viewImage(img)
		return img

	def processRawImage(self, image_file):
		'''Process the raw image to desired color and resolution status'''
		img = cv2.imread(image_file)
		img = img / np.max(img)
		img = img - np.mean(img)
		height, width = img.shape[:2]
		img = cv2.resize(img, (int(width/self.img_ratio), int(height/self.img_ratio)), fx=0, fy=0)
		return img

	def getBatchImages(self, train=1):
		'''Return a batch of stacked training OR testing images'''
		# If the batch is to be extracted from train OR test images list
		total_images = self.total_train_images if train else self.total_test_images
		local_pointer = self.last_image_pointer["train"] if train else self.last_image_pointer["test"]
		stacked_image_batch = []

		# Extracting images of batch_size length using the last pointer in the list
		img_lower_index = local_pointer
		img_higher_index = local_pointer + self.batch_size*self.time_step
		for index in range(img_lower_index, img_higher_index, self.time_step):
			# print (img_lower_index, img_higher_index, self.time_step, index)
			if ((index+1) < total_images):
				if train:
					stacked_image = np.concatenate([self.getTrainImage(index), self.getTrainImage(index+1)], -1)
				else:
					stacked_image = np.concatenate([self.getTestImage(index), self.getTestImage(index+1)], -1)					
				stacked_image_batch.append(stacked_image)
			else:
				print("Reached the end of the dataset")

		# Store the last batch pointer
		if train:
			self.last_image_pointer["train"] = self.last_image_pointer["train"] + self.batch_size 
		else:
			self.last_image_pointer["test"] = self.last_image_pointer["test"] + self.batch_size

		if self.debug_verbosity:
			print ("Stacked Image Batch Size: [%d]" %len(stacked_image_batch))
		return stacked_image_batch

	def viewImage(self, image_file):
		'''View the image depending upon the verbosity'''
		cv2.imshow("ImageWindow", image_file)
		cv2.waitKey()

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

	def defineCNNModel(self):
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
		self.getBatchImages()


if __name__ == '__main__':
	cnn_model_params = [3]
	rnn_model_params = [10]
	path_to_poses = "../dataset_images"	
	image_sequences = [["00"], ["11"]]
	img_ratio, image_verbosity, debug_verbosity = 0, 1, 1
	train_image_start_pointer, test_image_start_pointer = 500, 21
	operation_flags = [debug_verbosity, [train_image_start_pointer, test_image_start_pointer]]
	data_params = [path_to_poses, image_sequences, img_ratio, image_verbosity]
	rcnn_object = RCNN(cnn_model_params, rnn_model_params, data_params, operation_flags)
	rcnn_object.execute()