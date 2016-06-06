# DeepOSM.org

A website to browse OpenStreetMap errors detected by DeepOSM.

## Run the Site Locally

This development setup for Django/postgres/docker setup is based on the [stock Docker Compose docs for Django/Postgres](https://docs.docker.com/compose/django/). 

Start Docker Quickstart Terminal on a Mac. Then:

    cd /PATH_TO_REPO/deeposm.org
    docker-compose up

Then, the site is live at your docker IP, similar to: http://192.168.99.100:8000/

## Deploy to Amazon Elastic Beanstalk (EBS)

Follow [this tutorial](https://realpython.com/blog/python/deploying-a-django-app-and-postgresql-to-aws-elastic-beanstalk/). It goes like:

Use the EBS command line tools to deploy:

    pip install awsebcli

Per tutorial, run

    eb create
    eb config

Per tutorial, if you search for the terms ‘WSGI’ in the file, and you should find a configuration section to change application.py to:

    website/wsgi.py
   
Then:

    eb deploy
