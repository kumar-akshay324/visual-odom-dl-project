run:
	@python src/vo_estimation.py $(CAMERA)

install-dependencies:
	sudo -H pip3 install transforms3d
poses:
	@python src/odometry.py