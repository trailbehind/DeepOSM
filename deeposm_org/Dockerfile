FROM python:2.7
ENV PYTHONUNBUFFERED 1
RUN mkdir /deeposmorg
WORKDIR /deeposmorg
ADD requirements.txt /deeposmorg/
ADD . /deeposmorg/
VOLUME /deeposmorg
ENV PYTHONPATH /deeposmorg:$PYTHONPATH
RUN pip install -r requirements.txt ; python manage.py runserver 0.0.0.0:8000
