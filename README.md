# Deep OSM

Detect roads and features in satellite imagery by training a convnet with OSM data.

Work in progress.

![Deep OSM Project](https://gaiagps.mybalsamiq.com/mockups/4278030.png?key=1e42f249214928d1fa7b17cf866401de0c2af867)

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
    export PYTHONPATH=$PYTHONPATH:./lib/global_map
# Road/Trail Detection

## Overview

Detect OpenStreetMap (OSM) ways (streets and trails) in satellite imagery. Train the neural net using MapQuest open imagery, and an OSM ways.

## Background

* [TensorFlow](https://www.tensorflow.org/) - using this for the deep learning
* Hinton on [using a neural network to do this](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.1679&rep=rep1&type=pdf) - best/recent paper on doing this, great success with these methods
* Links from the Tensorflow site
    * [MNIST Data and Background](http://yann.lecun.com/exdb/mnist/)
    * all the other links to Nielsen’s book and [Colah’s blog](http://colah.github.io/posts/2015-08-Backprop/)
* Deep Background
    * [original Information Theory paper by Shannon](http://worrydream.com/refs/Shannon%20-%20A%20Mathematical%20Theory%20of%20Communication.pdf)


## Methodology

Multilayer Convolutional Network

### Train

* Download a chunk of satellite imagery from MapQuest at max resolution, in 256x256px PNGs
* Download ways (i.e. road/trails) for that area from OSM 
* Generate training and evaluation data

### Test 

* Download a different set of training imagery and OSM ways, and see if we can predict the ways from the imagery

## Marshal Test Data

### MapZen vector gepjson tiles are convenient

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