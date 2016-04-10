

.PHONY: dev build

help:
	@echo 'Makefile for Deep-OSM/naipreader'
	@echo ''
	@echo 'make dev        Build and run /bin/bash'

IMAGE_NAME = deep-osm

build:
	docker build -t $(IMAGE_NAME) .

dev: build
	docker run -v `pwd`:/Deep-OSM -e CPLUS_INCLUDE_PATH=/usr/include/gdal -e C_INCLUDE_PATH=/usr/include/gdal -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) -ti $(IMAGE_NAME) /bin/bash


