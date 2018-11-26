import numpy as np
from transform3d import *

class Odometry:

	def __init__(self, cnn_model_params, rnn_model_params, data_params):
		'''
		Input: 	data_params 		= 	[path_to_dataset, image_sequences, pose_verbosity]
										image_sequences = [training sequences list, test sequences list]
				cnn_model_params	= 	[batch_size]
		'''
		self.batch_size = cnn_model_params[0]
		self.image_sequences = data_params[0]
		self.pose_verbostiy = data_params[-1]
		self.time_step = rnn_model_params[0]

		# Initialize and store all the poses of the concerned sequences together
		self.initTrainPoses()

	def rotMatToEulerVector(self, R):
		return euler_angle = mat2euler(R)

	def groundTruthPoses(self):
		current_dir =os.getcwd()
		self.train_pose_files = [current_dir + "/" + self.path_to_images + "/poses/" + str(index) + ".txt" for index in self.image_sequences[0]]

	def initTrainPoses(self):
		self.groundTruthPoses()
		self.translation_vectors = []
		self.euler_angle_vectors = []
		try:
			if not (self.train_pose_files == []):
				for pose_file in self.train_pose_files:
					try:
						with open(pose_file, "r") as sequence_pose_file:
							for line in sequence_pose_file:
								if not line==None:
									matrix_elements = line.split(" ")
									tf_matrix = np.asarray(matrix_elements)
									tf_matrix = np.reshape(3,4)
									rotation_matrix = tf_matrix[0:3:1, 0:3:1]
									euler_angles = rotMatToEulerVector(rotation_matrix)
									trans_vec = tf_matrix[0:4:1, 3:4:1]
									self.translation_vectors.append(trans_vec)
									self.euler_angle_vectors.append(euler_angles)
						if self.pose_verbostiy:
							print ("Sample Euler Angle Vector: %s" %str(self.translation_vectors[-1]))
							print ("Sample Translation Vector: %s" %str(self.euler_angle_vectors[-1]))
					except: FileNotFoundError
						print ("File Not Found %s" %(pose_file))
		except:
			print ("Error in parsing pose files")

	def translationVector(self, pointer):
		return self.translation_vectors[pointer]

	def eulerVector(self, pointer):
		return self.euler_angle_vectors[pointer]

	def getBatchPoses(self, pointer):
		# Extracting poses of batch_size length using the last pointer in the list
		for index in range(pointer, self.batch_size + pointer, self.time_step):
			if ((index+1) < len(image_list)):
				stacked_image = np.concatenate([image_list[index], image_list[index+1]], -1)
				stacked_image_batch.append(stacked_image)

