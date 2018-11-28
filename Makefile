run:
	@python src/process_images.py $(CAMERA)

install-dependencies:
	sudo -H pip3 install transforms3d
poses:
	@python src/odometry.py