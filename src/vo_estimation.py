import itertools
import matplotlib.pyplot as plt
import numpy as np
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

	def __init__(self, data):
		self.data = data

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
		# First Image as a Keras Tensor
		model_image_1 = Sequential()
		model_image_1.add(TimeDistributed(Conv2D(32, (3,3), padding="valid", input_shape=(channels, height, width, batches))))
		model_image_1.add(Activation("relu"))
		model_image_1.add(TimeDistributed(MaxPooling2D(pool_size=(2,2))))
		model_image_1.add(Dropout(0.1))
		# Second Image as a Keras Tensor
		model_image_2 = Input((size_axis_1, size_axis_2, size_axis_color))
		# Stacked Images together
		stacked_images = Concantenate([input_image_1, input_image_2], axis=-1)

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

