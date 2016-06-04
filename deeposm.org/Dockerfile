FROM python:2.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /deeposm.org
WORKDIR /deeposm.org
ADD requirements.txt /deeposm.org/
RUN pip install -r requirements.txt
ADD . /deeposm.org/
