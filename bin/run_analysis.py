#!/usr/bin/env python

import argparse
import json
import numpy
import pickle
import time

from src.run_analysis import analyze
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
                        default=96,
                        type=int,
                        help="around 100 is a good choice, defaults to 96 because cifar10 does")
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
    print("LOADING DATA: reading from disk and unpickling")
    t0 = time.time()
    with open(CACHE_PATH + 'training_images.pickle', 'r') as infile:
        training_images = pickle.load(infile)
    with open(CACHE_PATH + 'training_labels.pickle', 'r') as infile:
        training_labels = pickle.load(infile)
    with open(CACHE_PATH + 'test_images.pickle', 'r') as infile:
        test_images = pickle.load(infile)
    with open(CACHE_PATH + 'test_labels.pickle', 'r') as infile:
        test_labels = pickle.load(infile)
    with open(CACHE_PATH + 'label_types.json', 'r') as infile:
        label_types = json.load(infile)
    with open(cache_path + 'onehot_training_labels.json', 'r') as infile:
        onehot_training_labels = json.load(infile)
    with open(cache_path + 'onehot_test_labels.json', 'r') as infile:
        onehot_test_labels = json.load(infile)

    print("DATA LOADED: time to unpickle/json test data {0:.1f}s".format(time.time() - t0))

    parser = create_parser()
    args = parser.parse_args()

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
