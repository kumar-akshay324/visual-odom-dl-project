import numpy as np
from transforms3d.euler import mat2euler
import os, time, math

class Odometry:

	def __init__(self, cnn_model_params, rnn_model_params, data_params):
		'''
		Input: 	data_params 		= 	[path_to_poses, image_sequences, pose_verbosity]
										image_sequences = [training sequences list]
				cnn_model_params	= 	[batch_size]
				rnn_model_params 	= 	[time_steps]
		'''
		self.batch_size = cnn_model_params[0]
		self.path_to_poses = data_params[0]
		self.image_sequences = data_params[1]
		self.pose_verbostiy = data_params[-1]
		self.time_step = rnn_model_params[0]

		# Initialize and store all the poses of the concerned sequences together
		self.initTrainPoses()

	def rotMatToEulerVector(self, R):
		euler_angle = mat2euler(R)
		return euler_angle

	def groundTruthPoses(self):
		current_dir =os.getcwd()
		self.train_pose_files = [current_dir + "/" + self.path_to_poses + "/poses/" + str(index) + ".txt" for index in self.image_sequences]

	def initTrainPoses(self):
		self.groundTruthPoses()
		self.translation_vectors = []
		self.euler_angle_vectors = []
		if not (self.train_pose_files == []):
			for pose_file in self.train_pose_files:
				try:
					with open(pose_file, "r") as sequence_pose_file:
						for index, line in enumerate(sequence_pose_file):
							if not line==None:
								matrix_elements = line.split()
								matrix_elements = [float(item) for item in matrix_elements]

								tf_matrix = np.asarray(matrix_elements)
								tf_matrix = tf_matrix.reshape(3,4)

								rotation_matrix = tf_matrix[0:3:1, 0:3:1]
								euler_angles = self.rotMatToEulerVector(rotation_matrix)
								trans_vec = tf_matrix[0:4:1, 3:4:1]

								self.translation_vectors.append(trans_vec)
								self.euler_angle_vectors.append(euler_angles)

							if self.pose_verbostiy and index==0:
								print (line)
								print ("\n----\n ")
								print ("Sample Euler Angle Vector: %s" %str(self.euler_angle_vectors[0]))
								print ("Sample Rotation Matrix: %s" %str(rotation_matrix))
								print ("Sample Translation Vector: %s" %str(self.translation_vectors[0]))
								print ("\n----\n")
								print ("Sample Euler Angle Vector: %s" %str(euler_angles))
								print ("Sample Rotation Matrix: %s" %str(rotation_matrix))
								print ("Sample Translation Vector: %s" %str(trans_vec))

				except FileNotFoundError:
					print ("File Not Found %s" %(pose_file))
				finally:
					if not len(self.translation_vectors) == len(self.euler_angle_vectors):
						print ("Unequal lengths of translation and euler vector lists")
					else:
						print ("Successfully read ground truth poses")

	def translationVectorList(self, pointer):
		return [self.translation_vectors[pointer], self.translation_vectors[pointer+1]]

	def eulerVectorList(self, pointer):
		return [self.euler_angle_vectors[pointer], self.euler_angle_vectors[pointer+1]]

	def getBatchPoses(self, pointer):
		'''	Input:
				pointer 		= Pointer to the line in the images/poses list to start extracting the poses
			Output:
				stacked_poses_batch 	= {"position": [], "orientation": [] }
											position: [[image_0 position list], [image_1 position list]]
											orientation: [[image_0 euler list], [image_1 euler list]]
				image_0 refers to first image and image_1 refers to second image
		'''
		stacked_poses_batch = {"position": [], "orientation": [] }
		# Extracting poses of batch_size length using the last pointer in the list
		for index in range(pointer, (self.batch_size * self.time_step) + pointer, self.time_step):
			if ((index+1) < len(self.translation_vectors)):
				stacked_poses_batch["position"].append(self.translationVectorList(index))
				stacked_poses_batch["orientation"].append(self.eulerVectorList(index))
			else:
				print ("Index: %d and Translation & Euler Vector lengths: %d & %d" %(index, len(translation_vectors), len(translation_vectors)))
				print ("End of dataset")
		if self.pose_verbostiy:
			print ("Final Batch of Poses: %s" %str(stacked_poses_batch))
		return stacked_poses_batch

	def execute(self, pointer):
		'''	Input:
				pointer 		= Pointer to the line in the images/poses list to start extracting the poses
		'''
		self.getBatchPoses(4539)

if __name__ == '__main__':
	cnn_model_params = [5]
	rnn_model_params = [10]
	path_to_poses = "../dataset_images"
	image_sequences = ["00", "01"]
	pose_verbostiy = 0
	data_params = [path_to_poses, image_sequences, pose_verbostiy]
	odom_object = Odometry(cnn_model_params, rnn_model_params, data_params)
	odom_object.execute(4539)
