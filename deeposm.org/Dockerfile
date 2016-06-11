FROM python:2.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /deeposmorg
WORKDIR /deeposmorg
ADD requirements.txt /deeposmorg/
RUN pip install -r requirements.txt
ADD . /deeposmorg/