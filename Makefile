

.PHONY: dev build

help:
	@echo 'Makefile for Deep-OSM/naipreader'
	@echo ''
	@echo 'make dev        Build and run /bin/bash'

IMAGE_NAME = deep-osm

build:
	docker build -t $(IMAGE_NAME) .

dev: build
	docker run -v `pwd`:/Deep-OSM -e CPLUS_INCLUDE_PATH=/usr/include/gdal -e C_INCLUDE_PATH=/usr/include/gdal -e AWS_ACCESS_KEY_ID="AKIAJW52XOFMBMZ7AQTQ" -e AWS_SECRET_ACCESS_KEY="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" -ti $(IMAGE_NAME) /bin/bash


