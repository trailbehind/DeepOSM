"""Create data for all states, and upload to deeposm.org's S3 bucket."""

import pickle

from src.naip_images import NAIPDownloader
from src.s3_client_deeposm import post_findings_to_s3
from src.single_layer_network import MODEL_METADATA_PATH, train_on_cached_data, load_model
from src.training_data import CACHE_PATH, METADATA_PATH, create_tiled_training_data, cache_paths


def main():
	  """DO each state one by one."""
    randomize_naips = False
    naip_year = 2013
    naip_states = ['de': ['http://download.geofabrik.de/north-america/us/delaware-latest.osm.pbf'], 
                   'me': ['http://download.geofabrik.de/north-america/us/maine-latest.osm.pbf']
                  ]
    number_of_naips = 25

    extract_type = 'highway'
    bands = [1, 1, 1, 1]
    tile_size = 64
    pixels_to_fatten_roads = 3
    tile_overlap = 1

    neural_net = 'one_layer_relu_conv'
    number_of_epochs = 25

    for state, filenames in naip_states:
        naiper = NAIPDownloader(number_of_naips, state, naip_year)
        raster_data_paths = naiper.download_naips()
        cache_paths(raster_data_paths)
        create_tiled_training_data(raster_data_paths, extract_type, bands, tile_size,
                                   pixels_to_fatten_roads, filenames,
                                   tile_overlap, state)

    with open(CACHE_PATH + METADATA_PATH, 'r') as infile:
        training_info = pickle.load(infile)

    test_images, model = train_on_cached_data(raster_data_paths, neural_net,
                                              training_info['bands'], training_info['tile_size'],
                                              number_of_epochs)

    with open(CACHE_PATH + MODEL_METADATA_PATH, 'r') as infile:
        model_info = pickle.load(infile)
    
    model = load_model(model_info['neural_net_type'], model_info['tile_size'], len(model_info['bands']))
    post_findings_to_s3(raster_data_paths, model, training_info)

