"""Create data for all states, and upload to deeposm.org's S3 bucket."""

import pickle
import requests

from src.s3_client_deeposm import post_findings_to_s3
from src.single_layer_network import train_on_cached_data
from src.training_data import CACHE_PATH, METADATA_PATH, download_and_serialize


def main():
    """Analyze each state and publish results to deeposm.org."""

    naip_year = 2013
    naip_states = {'de': ['http://download.geofabrik.de/north-america/us/delaware-latest.osm.pbf'],
                   'ia': ['http://download.geofabrik.de/north-america/us/iowa-latest.osm.pbf'],
                   'me': ['http://download.geofabrik.de/north-america/us/maine-latest.osm.pbf']
                   }
    number_of_naips = 175

    extract_type = 'highway'
    bands = [1, 1, 1, 1]
    tile_size = 64
    pixels_to_fatten_roads = 3
    tile_overlap = 1

    neural_net = 'two_layer_relu_conv'
    number_of_epochs = 10
    randomize_naips = False

    for state in naip_states:
        filenames = naip_states[state]
        raster_data_paths = download_and_serialize(number_of_naips,
                                                   randomize_naips,
                                                   state,
                                                   naip_year,
                                                   extract_type,
                                                   bands,
                                                   tile_size,
                                                   pixels_to_fatten_roads,
                                                   filenames,
                                                   tile_overlap)
        model = train_on_cached_data(neural_net, number_of_epochs)
        with open(CACHE_PATH + METADATA_PATH, 'r') as infile:
            training_info = pickle.load(infile)
        post_findings_to_s3(raster_data_paths, model, training_info, training_info['bands'], False)

    requests.get('http://www.deeposm.org/refresh_findings/')


if __name__ == "__main__":
    main()
