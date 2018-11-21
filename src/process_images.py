"""Using PyKITTI for odometry"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti

import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import mnist

from keras.layers.convolutional import *
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import *

from keras import backend as K

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.optimizers import *
import pandas as pd

from keras import backend as K
K.set_image_dim_ordering('th')

class ProcessImages():

	def __init__(self, camera_side, path_to_images):
		self.camera_side = camera_side
		self.raw_path = os.getcwd() + "/" + path_to_images
		print ("Raw Dataset Path: %s" %self.raw_path)
		self.path_to_images = path_to_images

	def completeImagesPath(self, sequence_id):

		if (sequence_id%10==sequence_id):
			self.sequence_id = ("0" + str(sequence_id))
		else:
			self.sequence_id = str(sequence_id)
		print  ("Final Sequence ID: %s" %self.sequence_id)
		
		folder_name = "image_0" if self.camera_side == "left" else "image_1"
		self.path_to_images = os.getcwd() + "/" + self.path_to_images + "sequences/"+ self.sequence_id + "/" + folder_name

		if not os.path.isdir(self.path_to_images):
			print ('\033[93m' +  "Image Sequence Folder does not exist: %s" %str(self.path_to_images))
			return False
		else:
			print ('\033[92m' +  "Image Sequence Folder exists: %s" %str(self.path_to_images))
			return True

	def loadImages(self):
		
		train_datagen = ImageDataGenerator()
		print ("Train Images Path: %s" %(self.path_to_images))
		
		self.train_batches = train_datagen.flow_from_directory(directory=self.train_path,
																target_size=(148, 148),
																color_mode="rgb",
																batch_size=self.size_batches)


	def loadImageDataset(self):
		'''Load the data. Optionally, specify the frame range to load'''

		# PyKITTI object for 
		self.image_dataset = pykitti.odometry(self.raw_path, self.sequence_id, frames=range(0, 20, 5))
		print ("Length of dataset: %s" %str(len(self.image_dataset)))

		# self.image_dataset.calib:      Calibration data are accessible as a named tuple
		# self.image_dataset.timestamps: Timestamps are parsed into a list of timedelta objects
		# self.image_dataset.poses:      List of ground truth poses T_w_cam0
		# self.image_dataset.camN:       Generator to load individual images from camera N
		# self.image_dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
		# self.image_dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
		# self.image_dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]



	def visualizeDataset(self):
		# Grab some data
		second_pose = self.image_dataset.poses[1]
		first_gray = next(iter(self.image_dataset.gray))
		first_cam1 = next(iter(self.image_dataset.cam1))
		first_rgb = self.image_dataset.get_rgb(0)
		first_cam2 = self.image_dataset.get_cam2(0)
		third_velo = self.image_dataset.get_velo(2)

		# Display some of the data
		print('\nSequence: ' + str(self.dataset.sequence))
		print('\nFrame range: ' + str(self.dataset.frames))

		print('\nGray stereo pair baseline [m]: ' + str(self.dataset.calib.b_gray))
		print('\nRGB stereo pair baseline [m]: ' + str(self.dataset.calib.b_rgb))

		print('\nFirst timestamp: ' + str(self.dataset.timestamps[0]))
		print('\nSecond ground truth pose:\n' + str(second_pose))

		# f, ax = plt.subplots(2, 2, figsize=(15, 5))
		# ax[0, 0].imshow(first_gray[0], cmap='gray')
		# ax[0, 0].set_title('Left Gray Image (cam0)')

		# ax[0, 1].imshow(first_cam1, cmap='gray')
		# ax[0, 1].set_title('Right Gray Image (cam1)')

		# ax[1, 0].imshow(first_cam2)
		# ax[1, 0].set_title('Left RGB Image (cam2)')

		# ax[1, 1].imshow(first_rgb[1])
		# ax[1, 1].set_title('Right RGB Image (cam3)')

		# f2 = plt.figure()
		# ax2 = f2.add_subplot(111, projection='3d')
		# # Plot every 100th point so things don't get too bogged down
		# velo_range = range(0, third_velo.shape[0], 100)
		# ax2.scatter(third_velo[velo_range, 0],
		#             third_velo[velo_range, 1],
		#             third_velo[velo_range, 2],
		#             c=third_velo[velo_range, 3],
		#             cmap='gray')
		# ax2.set_title('Third Velodyne scan (subsampled)')

		# plt.show()

	def execute(self):
		self.completeImagesPath(0)
		self.loadImageDataset()
		# self.visualizeDataset()

if __name__ == '__main__':
	camera_side = "left"
	path_to_images = "../dataset_images/"
	image_object = ProcessImages(camera_side, path_to_images)
	image_object.execute()