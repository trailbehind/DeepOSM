# DeepOSM

Detect roads and features in satellite imagery, by training neural networks with OpenStreetMap (OSM) data. The gist:

* Download a chunk of satellite imagery
* Download OSM data that shows roads/features for that area
* Generate training and evaluation data

Read below to run the code. [I am blogging my work journal too](http://trailbehind.github.io/DeepOSM/). 

Contributions are welcome. Open an issue if you want to discuss something to do, or [email me](mailto:andrew@gaiagps.com).

## Background on Data - NAIPs and OSM PBF

For training data, DeepOSM cuts tiles out of [NAIP images](http://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/), which provide 1 meter per pixel resolution, with RGB+infrared data bands.

For training labels, DeepOSM uses PBF extracts of OSM data, which contain features/ways in binary format, which can be munged with Python.

The [NAIPs come from a requester pays bucket on S3 set up by Mapbox](http://www.slideshare.net/AmazonWebServices/open-data-innovation-building-on-open-data-sets-for-innovative-applications), and the OSM extracts come [from geofabrik](http://download.geofabrik.de/).

## Install Requirements

### AWS Credentials

You need AWS credentials to download NAIPs from an S3 requester-pays bucket. This only costs a few cents for a bunch of images, but you need a credit card on file.

 * get your [AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from AWS](http://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html)

 * export them as environment variables (and maybe add to your bash or zprofile)

    export AWS_ACCESS_KEY_ID='FOO'

    export AWS_SECRET_ACCESS_KEY='BAR'

### Install Docker

First, [install a Docker Binary](https://docs.docker.com/engine/installation/).

I also needed to set my VirtualBox default memory to 12GB. libosmium needed 4GB, and the neural net needed even more. This is easy:

 * start Docker, per the install instructions
 * stop Docker
 * open VirtualBox, and increase the memory of the VM Docker made

### Run Scripts

Start Docker, then run:

```bash
make dev
```

### Download NAIP, PBF, and Analyze

Inside Docker, the following Python script will work. It will download all source data, tile it into training/test data and labels, train the neural net, and generate image and text output.

    python src/run_analysis.py

This will download four NAIPs, and tile it into NxNx1 bands of data (IR band). Then it will download some PBF files and extract the ways for the NAIPs.

It will produce PNGs of the ways, labels, and predictions overlaid on the tiff. It will be able to guess with 69% accuracy if a 64x64px tiles contains highways.

![NAIP with Ways and Predictions](https://pbs.twimg.com/media/Cg2F_tBUcAA-wHs.png)

### Jupyter Notebook

Alternately, development/research can be done via jupyter notebooks:

```bash
make notebook
```

To access the notebook via a browser on your host machine, find the IP VirtualBox is giving your default docker container by running:

```bash
docker-machine ls

NAME      ACTIVE   DRIVER       STATE     URL                         SWARM   DOCKER    ERRORS
default   *        virtualbox   Running   tcp://192.168.99.100:2376           v1.10.3
```

The notebook server is accessible via port 8888, so in this case you'd go to:
http://192.168.99.100:8888

### Readings

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

### Original Idea

This was the general idea to start, and working with TMS tiles sort of worked (see first 50 or so commits), so DeepOSM got switched to better data:

![Deep OSM Project](https://gaiagps.mybalsamiq.com/mockups/4278030.png?key=1e42f249214928d1fa7b17cf866401de0c2af867)