run:
	@python src/process_images.py $(CAMERA)

install-dependencies:
	sudo -H pip install transforms3d
