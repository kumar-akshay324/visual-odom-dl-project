# Visual SLAM - Visual Odometry Estimation using Deep Learning on the KITTI Dataset
Course Project for Deep Learning for Robot Perception

Implementation of Visual SLAM concepts for course Deep Learning for Robot Perception. 

## Dependencies
	
* PyKITTI - [https://github.com/utiasSTARS/pykitti](https://github.com/utiasSTARS/pykitti)
* Numpy
* Keras

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

## Usage

The complete implementation is to be run using the _make_ filesystem with commands inside the _Makefile_

* **odometry.py** 			- Extract the poses for corresponding stacked images as a stacked batch
* **vo_estimation.py**  	- Process the stacked images and run the DL model on it
