# DeepOSM.org

A website to browse OpenStreetMap errors detected by DeepOSM, using Django, Postgres, and Docker.

This Django/postgres/docker setup is based on the [stock Docker Compose docs for Django/Postgres](https://docs.docker.com/compose/django/). 

The [deeposm.org Docker image is on DockerHub](https://hub.docker.com/r/deeposm/deeposm.org/).


## Run the Site Locally

Start Docker Quickstart Terminal on a Mac. Then:

    cd /PATH_TO_REPO/deeposm.org
    docker-compose -f deeposm-dev.yml up

Then, the site is live at your docker IP. Mine is: http://192.168.99.100:8000/

## Deploy to Amazon Elastic Beanstalk (EBS)

I followed [this tutorial](https://realpython.com/blog/python/deploying-a-django-app-and-postgresql-to-aws-elastic-beanstalk/).

Use the EBS command line tools to deploy:

    pip install awsebcli

Per tutorial, run

    eb create
    eb config

Per tutorial, if you search for the terms ‘WSGI’ in the file, and you should find a configuration section to change application.py to:

    website/wsgi.py
   
Then:

    eb deploy
