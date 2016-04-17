# Deep OSM

Detect roads and features in satellite imagery, by training neural networks with OpenStreetMap (OSM) data. The gist:

* Download a chunk of satellite imagery
* Download ways (i.e. road/trails) for that area from OSM
* Generate training and evaluation data

This is a work in progress. Experiment 1 went well, and now the goal is better data Experiment 2. Read below to run the code for either.

Contributions are welcome. Open an issue if you want to discuss something to do, or [email me](mailto:andrew@gaiagps.com).

# Experiment 2 - NAIPs and OSM PBF

## Overview

For the experiment 2, I decided to tile/clip [NAIP images](http://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) instead of using Mapquest imagery tiles, based on things I heard from a couple of people:

* the IR bands are supposed to be important for the analysis
* the images are higher resolution
* the images aren't smashed into spherical mercator, which might matter at the poles?

When I switched to NAIPs, it no longer made sense to use pre-sliced vector tiles for the training labels. Instead, I'm extracting OSM data from PBF extracts, and manually clipping them to match arbitrary NAIP tiles.

I am currently working on this stage. The [NAIPs come from a requester pays bucket on S3 set up by Mapbox](http://www.slideshare.net/AmazonWebServices/open-data-innovation-building-on-open-data-sets-for-innovative-applications), and the OSM extracts come [from geofabrik](http://download.geofabrik.de/).

## Install Requirements

### AWS Credentials

You need AWS credentials to download NAIPs from an S3 requester-pays bucket.

 * get your [AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from AWS](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

 * export them as environment variables (and maybe add to your bash or zprofile)


    export AWS_ACCESS_KEY_ID='FOO'

    export AWS_SECRET_ACCESS_KEY='BAR'

### Install Docker

I also needed to set my VirtualBox default memory to 4GB, from the default 1GB, to get libosmium to work.

Then, I needed to up it again, to 16 GB, to get the neural net to work.

## Run Scripts

Start Docker, then run:

```bash
make dev
```

Then the following Python script will work in the Docker machine.

Alternately, development/research can be done via jupyter notebooks:

```bash
make notebook
```

To access the notebook via a browser on your host machine, find the IP VirtualBox is giving your
default docker container by running:

```bash
docker-machine ls

NAME      ACTIVE   DRIVER       STATE     URL                         SWARM   DOCKER    ERRORS
default   *        virtualbox   Running   tcp://192.168.99.100:2376           v1.10.3
```

The notebook server is accessible via port 8888, so in this case you'd go to:
http://192.168.99.100:8888

## Download NAIP, PBF, and Analyze

    python data_pipeline/rasterize_naips.py --use_pbf_cache=True --render_results=True

This will download one NAIP, and tile it into cubes (NxNx4 bands of data). Then it will download a PBF file and extract the ways for the NAIP.

It will produce a JPEG of the ways, labels, and predictions overlaid on the tiff.

![NAIP with Ways](https://pbs.twimg.com/media/Cft3GbeUkAAqqAd.jpg)

# Experiment 1 - TMS Tiles

## Overview

I first trained on a set of Mapzen vector tiles, which conveniently map onto Mapquest imagery tiles. This was the simplest possible thing I thought to do... this data jammed right into the [Tensorflow](http://tensorflow.org) tutorials.

The label_chunks_cnn.py script seemed to be able to guess whether a 256px tile at some zooms has an OSM way on it with ~70% accuracy for very small image sets. The label_chunks_softmax.py was worse. There is also some chance the output was just random and buggy too.

## Install Requirements

This has been run on OSX Yosemite (10.10.5).

    brew install libjpeg
    pip3 install -r requirements.txt
    sudo easy_install --upgrade six

Install globalmaptiles.py

    mkdir lib
    cd lib
    git clone git@gist.github.com:d5bf14750eff1197e8c5.git global_map
    cd ..
    export PYTHONPATH=$PYTHONPATH:./lib/global_map:./data_pipeline

## Run the Script
This will download vectors, imagery, and run the analysis.

    python3 analysis/label_chunks_softmax.py download-data MAPZEN_KEY
    python3 analysis/label_chunks_softmax.py train

This will use a convolutional neural network (CNN) from TensorFlow tutorial 2, instead of the softmax from tutorial 1.

    python3 analysis/label_chunks_cnn.py train

# Background

This was the general idea to start:

![Deep OSM Project](https://gaiagps.mybalsamiq.com/mockups/4278030.png?key=1e42f249214928d1fa7b17cf866401de0c2af867)

## Background

* [TensorFlow](https://www.tensorflow.org/) - using this for the deep learning, do multilayer, deep CNN
* [Learning to Detect Roads in High-Resolution Aerial
Images](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.1679&rep=rep1&type=pdf) - best/recent paper on doing this, great success with these methods
* Similar Efforts with OSM Data
    * [OSM-Crosswalk-Detection](https://github.com/geometalab/OSM-Crosswalk-Detection) - uses Keras to detect crosswalks, a class project (Fall 2015)
    * [OSM-HOT-ConvNet](https://github.com/larsroemheld/OSM-HOT-ConvNet) - attempted use for disaster response, author thinks it's only 69% accurate at pixel level (Fall 2016)
* [Parsing Natural Scenes and Natural Language
with Recursive Neural Networks (RNNs)](http://ai.stanford.edu/~ang/papers/icml11-ParsingWithRecursiveNeuralNetworks.pdf)
* Links from the Tensorflow site
    * [MNIST Data and Background](http://yann.lecun.com/exdb/mnist/)
    * all the other links to Nielsen’s book and [Colah’s blog](http://colah.github.io/posts/2015-08-Backprop/)
* Deep Background
    * [original Information Theory paper by Shannon](http://worrydream.com/refs/Shannon%20-%20A%20Mathematical%20Theory%20of%20Communication.pdf)
