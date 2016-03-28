# Deep OSM

Detect roads and features in satellite imagery by training a convnet with OSM data.

# Work in progress

At this point, the label_chunks_cnn.py script seems to be able to guess whether a 256px tile at some zooms has an OSM way on it with ~70% accuracy for very small image sets.

# Install Requirements

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

# Run the Script
This will download vectors, imagery, and run the analysis.

    python3 label_chunks_softmax.py download-data MAPZEN_KEY
    python3 label_chunks_softmax.py train

This will use a convolutional neural network (CNN) from TensorFlow tutorial 2, instead of the softmax from tutorial 1.

    python3 label_chunks_cnn.py train

# Download NAIP Imagery

I just started the script to download NAIPs. You need AWS credentials to download NAIPs from an S3 requester-pays bucket.

 * set you [AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or otherwise authenticate with AWS](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)
 
Then run:

    python3 data_pipeline/download_naips.py download

This will download one NAIP. Have fun!

# Contributions

Contributions are welcome. Open an issue if you want to discuss something to do.

Things I have thought about working on next include:

 * run the CNN on Google Cloud or AWS, so it can use parallel GPUs
 * change the CNN to do pixel-level identification, instead of one-hotting tiles
 * download and process NAIP imagery, instead of JPG tiles
 * do features like tennis courts or baseball diamonds maybe (does OSM have enough training data?)
 * lots of data pipeline work, and an intense bit of neural net work

# Road/Trail Detection Project Idea

## Overview

Detect OpenStreetMap (OSM) ways (streets and trails) in satellite imagery. Train the neural net using MapQuest open imagery, and an OSM ways.

![Deep OSM Project](https://gaiagps.mybalsamiq.com/mockups/4278030.png?key=1e42f249214928d1fa7b17cf866401de0c2af867)

## Background

* [TensorFlow](https://www.tensorflow.org/) - using this for the deep learning, do multilayer, deep CNN
* [Learning to Detect Roads in High-Resolution Aerial
Images](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.1679&rep=rep1&type=pdf) - best/recent paper on doing this, great success with these methods
* [Parsing Natural Scenes and Natural Language
with Recursive Neural Networks (RNNs)](http://ai.stanford.edu/~ang/papers/icml11-ParsingWithRecursiveNeuralNetworks.pdf)
* Links from the Tensorflow site
    * [MNIST Data and Background](http://yann.lecun.com/exdb/mnist/)
    * all the other links to Nielsen’s book and [Colah’s blog](http://colah.github.io/posts/2015-08-Backprop/)
* Deep Background
    * [original Information Theory paper by Shannon](http://worrydream.com/refs/Shannon%20-%20A%20Mathematical%20Theory%20of%20Communication.pdf)

### Train

* Download a chunk of satellite imagery from MapQuest at max resolution, in 256x256px PNGs
* Download ways (i.e. road/trails) for that area from OSM 
* Generate training and evaluation data

### Test 

* Download a different set of training imagery and OSM ways, and see if we can predict the ways from the imagery

## Marshal Test Data

### Mapzen vector gepjson tiles are convenient

* flatten it into matrices of trail/no trail
* check tiles visually 

### download imagery data

* use GDAL to download and composite image tiles

## Scale Up

Data Size

* do a tiny area and do it all locally for testing
* use one or more GPUs on Amazon if bottlenecked

Download OSM data, parse out ways

* alternative to Clipper method: load it into Postgres and do it that way

Accuracy

* mimic Hinton’s methods, esp. for getting real road geometries
* see if we can identify trails nearly as well as roads
