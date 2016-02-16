# Learning Project

There's nothing to see here, move along. None of this code probably works. Still working on tutorials to learn deep learning, so I can apply to OSM and satellite imagery.

# Install Requirements

    pip install -r requirements.txt 

Install globalmaptiles.py

    git clone git@gist.github.com:1193577.git

For using the MNIST data, clone the tensorflow repo, and add the mnist example to your PYTHONPATH:

    export PYTHONPATH=$PYTHONPATH:/PATH_TO_REPO/tensorflow/tensorflow/examples/tutorials/mnist/:/PATH_TO_REPO/lib/global_map

# Trail Detection

## Overview

Detect OpenStreetMap (OSM) ways (streets and trails) in satellite imagery. Train the neural net using MapQuest open imagery, and an OSM extract of San Francisco

## Background

* [TensorFlow](https://www.tensorflow.org/) - using this for the deep learning
* Hinton on [using a neural network to do this](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.1679&rep=rep1&type=pdf) - best/recent paper on doing this, great success with these methods
* Links from the Tensorflow site
    * [MNIST Data and Background](http://yann.lecun.com/exdb/mnist/)
    * all the other links to Nielsen’s book and [Colah’s blog](http://colah.github.io/posts/2015-08-Backprop/)
* Deep Background
    * [original Information Theory paper by Shannon](http://worrydream.com/refs/Shannon%20-%20A%20Mathematical%20Theory%20of%20Communication.pdf)



## Methodology

Multilayer Convolutional Network, softmax, same as MNIST TensorFlow example to start

### Train

* Download a chunk of satellite imagery from MapQuest at max resolution, in 256x256px PNGs
* Download ways (i.e. road/trails) for that area from OSM
* Generate training and evaluation data in the same form as the MNIST data

### Test 

* Download a different set of training imagery and OSM ways, and see if we can predict the ways from the imagery



## **Marshal San Francisco Test Data**

### download [MapZen SF extract](https://mapzen.com/data/metro-extracts/)

* clip it into squares that align with image tiles
    * write some Python to do the clipping, using clipper, 
* then flatten it into matrices of trail/no trail

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