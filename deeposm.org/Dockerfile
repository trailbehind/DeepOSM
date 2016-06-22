FROM python:3.4
ENV PYTHONUNBUFFERED 1
RUN mkdir /deeposmorg
WORKDIR /deeposmorg
ADD requirements.txt /deeposmorg/
RUN pip install -r requirements.txt
ADD . /deeposmorg/