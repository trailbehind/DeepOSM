# DeepOSM.org

A website to browse OpenStreetMap errors detected by DeepOSM, using Django, Postgres, and Docker.

This Django/postgres/docker setup is based on the [stock Docker Compose docs for Django/Postgres](https://docs.docker.com/compose/django/). 

The [deeposm.org Docker image is on DockerHub](https://hub.docker.com/r/deeposm/deeposm.org/).


## Run the Site Locally

Start Docker Quickstart Terminal on a Mac. Then:

    cd /PATH_TO_REPO/deeposm.org
    docker-compose -f deeposm-dev.yml up

Then, the site is live at your docker IP. Mine is: http://192.168.99.100:8000/

## Deploy to Amazon EC2 Container Service (ECS)

Use the Amazon ecs-cli tool to deploy from command line. I followed [this guide](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/ECS_CLI_tutorial.html).

These commands will work, assuming you have already:

1. made an AWS account
2. stored your credentials in environment variables like you would have using DeepOSM
3. made a keypair in the AWS console (which you'll put the name of in the 2nd command below)

Make the cluster:

    ecs-cli configure --region us-east-1 --access-key $AWS_ACCESS_KEY_ID --secret-key $AWS_SECRET_ACCESS_KEY --cluster deeposm-site
    ecs-cli up --keypair KEYPAIR_NAME_IN_AWS_CONSOLE  --capability-iam --size 2 --instance-type t2.medium 

ecs-cli doesn't support environment variables, so this swizzles the vars with sed, instead of doing it right (using an aws cli config file I believe):

    sed -e "s#AWS_SECRET_ACCESS_KEY#AWS_SECRET_ACCESS_KEY='$AWS_SECRET_ACCESS_KEY'#g" -e "s/AWS_ACCESS_KEY_ID/AWS_ACCESS_KEY_ID='$AWS_ACCESS_KEY_ID'/" deeposm-prod.yml > tmp-compose.yml

Bring up the Docker composed cluster, which includes web and database server:

    ecs-cli compose --file tmp-compose.yml service up

You might want to delete the tm yml:

    rm tmp-compose.yml