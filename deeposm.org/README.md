# DeepOSM.org

A website to browse OpenStreetMap errors detected by DeepOSM, using Django, Postgres, and Docker.

## Run the Site Locally

Start Docker Quickstart Terminal on a Mac.

    cd /PATH_TO_REPO/deeposm.org
    docker-compose up

Then, the site is live at your docker IP. Mine is: http://192.168.99.100:8000/

## Based On

This Django/postgres/docker setup is based on the [stock Docker Compose docs for Django/Postgres](https://docs.docker.com/compose/django/).

That didn't work out of the box for me though. I debugged for an hour, then googled for "devfs No space left on device postgres" becaue I was grasping at straws, and [found a way to clear out DDocker network cruft](https://github.com/docker/machine/issues/1779#issuecomment-136205807).