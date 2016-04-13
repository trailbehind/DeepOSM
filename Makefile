# Usage:
#    make dev              build and run /bin/bash in container.
#    make notebook         build and run jupyter notebook server.

.PHONY: dev notebook build

help:
	@echo 'Makefile for Deep-OSM'
	@echo ''
	@echo 'make dev             build and run /bin/bash'
	@echo 'make notebook        build and run jupyter notebook server'

IMAGE_NAME = deep-osm

build:
	docker build -t $(IMAGE_NAME) .

dev: build
	docker run -v `pwd`:/Deep-OSM \
               -w /Deep-OSM \
               -e CPLUS_INCLUDE_PATH=/usr/include/gdal \
               -e C_INCLUDE_PATH=/usr/include/gdal \
               -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
               -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
               -it $(IMAGE_NAME) /bin/bash

notebook: build
	docker run -p 8888:8888 \
               -e AWS_ACCESS_KEY_ID=$(AWS_ACCESS_KEY_ID) \
               -e AWS_SECRET_ACCESS_KEY=$(AWS_SECRET_ACCESS_KEY) \
               -v `pwd`:/Deep-OSM \
               -it $(IMAGE_NAME) /run_jupyter.sh
