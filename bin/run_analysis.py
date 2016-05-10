#!/usr/bin/env python

import argparse
import json
import numpy
import pickle
import time

from src.run_analysis import analyze
from src.create_training_data import load_data_from_disk
from src.render_results import render_results_for_analysis
from config_data import CACHE_PATH

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size",
                        default=64,
                        type=int,
                        help="tile the NAIP and training data into NxN tiles with this dimension")
    parser.add_argument("--training-batches",
                        default=100,
                        type=int,
                        help="the number of batches to train the neural net. 100 is good for a dry "
                        "run, but around 5000 is recommended for statistical significance")
    parser.add_argument("--batch-size",
                        default=100,
                        type=int,
                        help="around 100 is a good choice")
    parser.add_argument("--band-list",
                        default=[0, 0, 0, 1],
                        nargs=4,
                        type=int,
                        help="specify which bands to activate (R  G  B  IR). default is "
                        "--bands 0 0 0 1 (which activates only the IR band)")
    parser.add_argument("--render-results",
                        default=True,
                        action='store_true',
                        help="output data/predictions to JPEG")
    parser.add_argument("--model",
                        default='mnist',
                        choices=['mnist', 'cifar10'],
                        help="the model to use")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    raster_data_paths, training_images, training_labels, test_images, test_labels, label_types, onehot_training_labels, onehot_test_labels = load_data_from_disk()
    predictions = analyze(onehot_training_labels, onehot_test_labels, test_labels, training_labels, test_images, training_images, label_types,
                          args.model, args.band_list, args.training_batches, args.batch_size,
                          args.tile_size)
    if args.render_results:
        render_results_for_analysis(raster_data_paths, 
                                    training_labels, 
                                    test_labels, 
                                    predictions, 
                                    args.band_list, 
                                    args.tile_size)

if __name__ == "__main__":
    main()
