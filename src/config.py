"""Configuration for where to cache/process data."""

import os
import pickle
import shutil

# the name of the S3 bucket to post findings to
FINDINGS_S3_BUCKET = 'deeposm'

# set in Dockerfile as env variable
GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR")

# where training data gets cached/retrieved
CACHE_PATH = GEO_DATA_DIR + '/generated/'

NAIP_DATA_DIR = os.path.join(GEO_DATA_DIR, "naip")
LABELS_DATA_DIR = os.path.join(CACHE_PATH, "way_bitmaps")

METADATA_PATH = 'training_metadata.pickle'
LABEL_CACHE_DIRECTORY = 'training_labels/'
IMAGE_CACHE_DIRECTORY = 'training_images/'


def cache_paths(raster_data_paths):
    """Cache a list of naip image paths, to pass on to the train_neural_net script."""
    with open(CACHE_PATH + 'raster_data_paths.pickle', 'w') as outfile:
        pickle.dump(raster_data_paths, outfile)


def create_cache_directories():
    """Cache a list of naip image paths, to pass on to the train_neural_net script."""
    shutil.rmtree(CACHE_PATH)
    shutil.rmtree(GEO_DATA_DIR + '/openstreetmap')
    try:
        os.mkdir(CACHE_PATH)
    except:
        pass
    try:
        os.mkdir(CACHE_PATH + 'way_bitmaps')
    except:
        pass
    try:
        os.mkdir(CACHE_PATH + LABEL_CACHE_DIRECTORY)
    except:
        pass
    try:
        os.mkdir(CACHE_PATH + IMAGE_CACHE_DIRECTORY)
    except:
        pass
    try:
        os.mkdir(GEO_DATA_DIR + '/openstreetmap')
    except:
        pass
