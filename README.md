# Visual SLAM - Visual Odometry Estimation using Deep Learning on the KITTI Dataset
Course Project for Deep Learning for Robot Perception

### THIS PROJECT IS STILL UNDER HEAVY DEVELOPMENT 

## Dependencies
	
* Numpy
* Keras
* Tensorflow(Ideal backend to go with the data_format used in Keras implementation)
* OpenCV
* transform3d

## Data Folder Structure

```
├── dataset
	├── poses
	│   ├── 00.txt
	│   ├── 01.txt
	│   ├── 02.txt
	│   ├── 03.txt
	│   ├── 04.txt
	│   ├── 05.txt
	│   ├── 06.txt
	│   ├── 07.txt
	│   ├── 08.txt
	│   ├── 09.txt
	│   └── 10.txt
	└── sequences
	    └── 00
	        ├── calib.txt
	        └── image_0
	        	├──00000.png
	        	└──00001.png
	        ├── image_1
	        └── times.txt
```

* Download the grayscale stereovision camera image dataset from here [http://www.cvlibs.net/download.php?file=data_odometry_gray.zip](http://www.cvlibs.net/download.php?file=data_odometry_gray.zip)
* Download the calibration data from here [http://www.cvlibs.net/download.php?file=data_odometry_calib.zip](http://www.cvlibs.net/download.php?file=data_odometry_calib.zip)
* Extract and merge the folders above together

#### Note: You should not use the claibration files provided by default inside the camera image dataset, but one after merging the calibration data into it.

## File Description

* **odometry.py** 			- Extract the poses for corresponding stacked images as a stacked batch
* **vo_estimation.py**  	- Process the stacked images and run the DL model on it
* **process_images.py**		- pyKITTI based implemention - Still under development

## Usage

The complete implementation is to be run using the _make_ filesystem with commands inside the _Makefile_

* Run `make install-dependencies` to install system dependencies like transforms3d (geometric transformations librart) [https://pypi.org/project/transforms3d/] and OpenCV (image processing library wrapper for Python)[https://pypi.org/project/opencv-python/]

* Run `make poses`  to obtain a sample of ground truth poses for the training dataset. 

The parameters to be modified in the script are:
		*	data_params 				= 	[path_to_poses, image_sequences, pose_verbosity]
				where image_sequences 	= 	[training sequences list]
		*	cnn_model_params			= 	[batch_size]
		*	rnn_model_params 			= 	[time_steps]

* Run `make run` to obtain a batch of training images. Further integration with training and testing model is in progress.

The parameters to be modified in the script are:
		* 	data_params 				= 	[path_to_dataset, image_sequences, image_ratio,, image_verbosity]
				where image_sequences 	= 	[training sequences list, test sequences list]
		*	rnn_model_params			= 	[time_step,	LSTM_nodes]
		*	cnn_model_params			= 	[batch_size, ]
		*	operation_flags				= 	[debug_verbosity, [train_image_start_pointer, test_image_start_pointer]]

## Authors

* Akshay Kumar (akumar5@wpi.edu)
* Samruddhi Kadam (spkadam@wpi.edu)

