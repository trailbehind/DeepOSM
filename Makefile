# Usage:
#    make dev              build and run /bin/bash in container.
#    make notebook         build and run jupyter notebook server.

.PHONY: dev notebook build

help:
	@echo 'Makefile for DeepOSM'
	@echo ''
	@echo 'make dev             build and run /bin/bash'
	@echo 'make notebook        build and run jupyter notebook server'

IMAGE_NAME = deeposm

Dockerfile.cpu: Dockerfile.base
	cp $< $@ 

build-cpu: Dockerfile.cpu
	docker build -f $< -t $(IMAGE_NAME):latest .

Dockerfile.gpu: Dockerfile.base
	sed 's|^FROM tensorflow/tensorflow:.*$$|\0-gpu|' $< > $@

build-gpu: Dockerfile.gpu
	docker build -f $< -t $(IMAGE_NAME):latest-gpu .

build: build-cpu

dev-cpu: build-cpu
	./docker_run.sh cpu

dev-gpu: build-gpu
	./docker_run.sh gpu

dev: dev-cpu

test: build-cpu
	./docker_run.sh cpu python -m unittest discover

update-deeposmorg: build-gpu
	./docker_run.sh gpu python bin/update_deeposm.org

notebook-cpu: build-cpu
	./docker_run.sh cpu ./run_jupyter.sh

notebook-gpu: build-gpu
	./docker_run.sh gpu ./run_jupyter.sh

notebook: notebook-cpu
