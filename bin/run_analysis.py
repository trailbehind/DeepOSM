#!/usr/bin/env python

import argparse
import pickle
import time
from src.run_analysis import analyze, render_results_as_images


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size", default=64, type=int,
                        help="tile the NAIP and training data into NxN tiles with this dimension")
    parser.add_argument("--training-batches", default=100, type=int,
                        help="the number of batches to train the neural net. 100 is good for a dry "
                        "run, but around 5000 is recommended for statistical significance")
    parser.add_argument("--batch-size", default=96, type=int,
                        help="around 100 is a good choice, defaults to 96 because cifar10 does")
    parser.add_argument("--band-list", default=[1, 1, 1, 1], nargs=4, type=int,
                        help="specify which bands to activate (R  G  B  IR). default is "
                        "--bands 1 1 1 1 (which activates all bands)")
    parser.add_argument("--render-results", action='store_true',
                        help="output data/predictions to JPEG")
    parser.add_argument("--model", default='mnist', choices=['mnist', 'cifar10'],
                        help="the model to use")
    return parser


def main():
    print("LOADING DATA: reading from disk and unpickling")
    t0 = time.time()
    cache_path = '/data/cache/'
    with open(cache_path + 'training_images.pickle', 'r') as infile:
        training_images = pickle.load(infile)
    with open(cache_path + 'training_labels.pickle', 'r') as infile:
        training_labels = pickle.load(infile)
    with open(cache_path + 'test_images.pickle', 'r') as infile:
        test_images = pickle.load(infile)
    with open(cache_path + 'test_labels.pickle', 'r') as infile:
        test_labels = pickle.load(infile)
    with open(cache_path + 'label_types.pickle', 'r') as infile:
        label_types = pickle.load(infile)
    print("DATA LOADED: time to unpickle test data {0:.1f}s".format(time.time() - t0))

    parser = create_parser()
    args = parser.parse_args()

    predictions = analyze(test_labels,
                          training_labels,
                          test_images,
                          training_images,
                          label_types,
                          args.model,
                          args.band_list,
                          args.training_batches,
                          args.batch_size,
                          args.tile_size)

    if args.render_results:
        raster_data_paths = None
        way_bitmap_npy = None
        with open(cache_path + 'raster_data_paths.pickle', 'r') as infile:
            raster_data_paths = pickle.load(infile)
        with open(cache_path + 'way_bitmap_npy.pickle', 'r') as infile:
            way_bitmap_npy = pickle.load(infile)

        render_results_as_images(raster_data_paths,
                                 training_labels,
                                 test_labels,
                                 predictions,
                                 way_bitmap_npy,
                                 args.band_list,
                                 args.tile_size)


if __name__ == "__main__":
    main()
