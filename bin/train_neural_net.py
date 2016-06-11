#!/usr/bin/env python

"""Train a neural network using OpenStreetMap labels and NAIP images."""

import argparse
import pickle

# src.training_data must be included before src.single_layer_network,
# in order to import PIL before TFLearn - or PIL errors tryig to save a JPEG
from src.training_data import CACHE_PATH, METADATA_PATH
from src.s3_client_deeposm import post_findings_to_s3
from src.single_layer_network import train_on_cached_data


def create_parser():
    """Create the argparse parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-net",
                        default='one_layer_relu',
                        choices=['one_layer_relu', 'one_layer_relu_conv'],
                        help="the neural network architecture to use")
    parser.add_argument("--number-of-epochs",
                        default=5,
                        type=int,
                        help="the number of epochs to batch the training data into")
    parser.add_argument("--render-results",
                        action='store_true',
                        help="output data/predictions to JPEG, in addition to normal JSON")
    parser.add_argument("--post-findings-to-s3",
                        action='store_true',
                        help="post predicted errors JSON to an S3 bucket")
    return parser


def main():
    """Use local data to train the neural net, probably made by bin/create_training_data.py."""
    parser = create_parser()
    args = parser.parse_args()
    with open(CACHE_PATH + 'raster_data_paths.pickle', 'r') as infile:
        raster_data_paths = pickle.load(infile)

    with open(CACHE_PATH + METADATA_PATH, 'r') as infile:
        training_info = pickle.load(infile)

    test_images, model = train_on_cached_data(raster_data_paths, args.neural_net,
                                              training_info['bands'], training_info['tile_size'],
                                              args.number_of_epochs)

    if post_findings_to_s3:
        post_findings_to_s3(raster_data_paths, model, training_info, args.render_results)


if __name__ == "__main__":
    main()
