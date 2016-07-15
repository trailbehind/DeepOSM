# Based on: https://github.com/tensorflow/tensorflow/blob/fe056f0b5e52db86766761f5e6446a89c1aa3938/tensorflow/tools/docker/docker_run_gpu.sh

set -e

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [ ! -d ${CUDA_HOME}/lib64 ]; then
  echo "Failed to locate CUDA libs at ${CUDA_HOME}/lib64."
  exit 1
fi

CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
DEVICES=$(\ls /dev/nvidia* | \
                    xargs -I{} echo '--device {}:{}')

if "${DEVICES}" = "" ; then
  echo "Failed to locate NVidia device(s). Did you want the non-GPU container?"
  exit 1
fi

export IMAGE_NAME=deeposm

if "$1" = "true"; then
  sh /home/andrew/.profile
  docker run $CUDA_SO $DEVICES \
                -v `pwd`:/DeepOSM \
                 -w /DeepOSM \
                 -e CPLUS_INCLUDE_PATH=/usr/include/gdal \
                 -e C_INCLUDE_PATH=/usr/include/gdal \
                 -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
                 -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
                 -t ${IMAGE_NAME} python bin/update_deeposmorg.py
else
  docker run $CUDA_SO $DEVICES \
              -v `pwd`:/DeepOSM \
               -w /DeepOSM \
               -e CPLUS_INCLUDE_PATH=/usr/include/gdal \
               -e C_INCLUDE_PATH=/usr/include/gdal \
               -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
               -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
               -it ${IMAGE_NAME} /bin/bash
fi