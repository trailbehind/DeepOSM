# https://hub.docker.com/r/homme/gdal/
FROM geodata/gdal

# geodata/gdal sets the user to noboby, so reset to root
USER root

# Based on https://github.com/GoogleCloudPlatform/python-docker/blob/master/Dockerfile
# Link above installs stuff for Python, virtualenv
# Also kludged in stuff for Osmium and Git
RUN apt-get -q update && \
 apt-get install --no-install-recommends -y -q \
   libbz2-dev python2.7 python2.7-dev cmake python-pip build-essential git mercurial \
   libffi-dev libssl-dev libxml2-dev \
   libxslt1-dev libpq-dev libmysqlclient-dev libcurl4-openssl-dev \
   libjpeg-dev zlib1g-dev libpng12-dev \
   gfortran libblas-dev liblapack-dev libatlas-dev libquadmath0 \
   libfreetype6-dev pkg-config swig \
   zlib1g-dev libshp-dev libsqlite3-dev \
   libgd2-xpm-dev libexpat1-dev libgeos-dev libgeos++-dev libxml2-dev \
   libsparsehash-dev libv8-dev libicu-dev libgdal1-dev \
   libprotobuf-dev protobuf-compiler devscripts debhelper \
   fakeroot doxygen libboost-dev libboost-all-dev git-core \
   && \
 apt-get clean

# copy requirements.txt and run pip to install all dependencies into the virtualenv.
ADD requirements_base.txt /DeepOSM/requirements_base.txt
RUN pip install -r /DeepOSM/requirements_base.txt
ADD requirements_cpu.txt /DeepOSM/requirements_cpu.txt
RUN pip install -r /DeepOSM/requirements_cpu.txt
RUN ln -s /home/vmagent/src /DeepOSM

# install libosmium and pyosmium bindings
RUN git clone https://github.com/osmcode/libosmium /libosmium
RUN cd /libosmium && mkdir build && cd build && cmake .. && make
RUN git clone https://github.com/osmcode/pyosmium.git /pyosmium
RUN cd /pyosmium && pwd && python setup.py install

# update PYTHONPATH
ENV PYTHONPATH /DeepOSM:/DeepOSM/src:$PYTHONPATH
ENV GEO_DATA_DIR /DeepOSM/data

# Jupyter has issues with being run directly:
#    https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script, and set up our jupyter config.
COPY run_jupyter.sh /
COPY jupyter_notebook_config.py /root/.jupyter/
EXPOSE 8888

# install s3cmd, used to ls the RequesterPays bucket 
RUN apt-get --no-install-recommends -y -q install wget
RUN wget http://netix.dl.sourceforge.net/project/s3tools/s3cmd/1.6.0/s3cmd-1.6.0.tar.gz && tar xvfz s3cmd-1.6.0.tar.gz && cd s3cmd-1.6.0 && python setup.py install

# copy s3cmd config defaults to docker, which will later be
# updated with AWS credentials by Python inside docker
COPY s3config-default /root/.s3cfg

# https://github.com/tflearn/tflearn/issues/55
# its different if we do AWS GPUs
RUN apt-get install libhdf5-dev

ADD . /DeepOSM
WORKDIR /DeepOSM
