#!/usr/bin/env python

'''
    train a neural network using OpenStreetMap labels and NAIP images
'''

import argparse

from src.analysis import analyze
from src.training_data import load_data_from_disk
from src.training_visualization import render_results_for_analysis


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size",
                        default=64,
                        type=int,
                        help="tile the NAIP and training data into NxN tiles with this dimension")
    parser.add_argument("--bands",
                        default=[0, 0, 0, 1],
                        nargs=4,
                        type=int,
                        help="specify which bands to activate (R  G  B  IR). default is "
                        "--bands 0 0 0 1 (which activates only the IR band)")
    parser.add_argument("--render-results",
                        action='store_true',
                        help="output data/predictions to JPEG")
    parser.add_argument("--number-of-epochs",
                        default=100,
                        type=int,
                        help="the number of epochs to batch the training data into")
    parser.add_argument("--neural-net",
                        default='one_layer_relu',
                        choices=['one_layer_relu', 'one_layer_relu_conv'],
                        help="the neural network architecture to use")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    raster_data_paths, training_images, training_labels, test_images, test_labels, label_types, \
        onehot_training_labels, onehot_test_labels = load_data_from_disk()
    predictions = analyze(onehot_training_labels, onehot_test_labels, test_labels, training_labels,
                          test_images, training_images, label_types, args.neural_net,
                          args.bands, args.tile_size, args.number_of_epochs)
    if args.render_results:
        render_results_for_analysis(raster_data_paths, training_labels, test_labels, predictions,
                                    args.band_list, args.tile_size)


if __name__ == "__main__":
    main()
