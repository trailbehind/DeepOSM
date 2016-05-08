#!/usr/bin/env python

import argparse
import json
import os
import pickle
import time

from src.download_naips import NAIPDownloader
from src.create_training_data import (NUMBER_OF_NAIPS, RANDOMIZE_NAIPS, NAIP_STATE, NAIP_RESOLUTION,
                                      NAIP_YEAR, NAIP_SPECTRUM, NAIP_GRID, HARDCODED_NAIP_LIST,
                                      random_training_data, equalize_data, split_train_test)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size",
                        default=64,
                        type=int,
                        help="tile the NAIP and training data into NxN tiles with this dimension")
    parser.add_argument("--band-list",
                        default=[0, 0, 0, 1],
                        nargs=4,
                        type=int,
                        help="specify which bands to activate (R  G  B  IR). default is "
                        "--bands 0 0 0 1 (which activates only the IR band)")
    parser.add_argument("--extract-type",
                        default='highway',
                        choices=['highway', 'tennis'],
                        help="the type of feature to identify")
    parser.add_argument("--save_clippings",
                        action='store_true',
                        help="save the training data tiles to /data/naip")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    raster_data_paths = NAIPDownloader(NUMBER_OF_NAIPS, RANDOMIZE_NAIPS, NAIP_STATE, NAIP_YEAR,
                                       NAIP_RESOLUTION, NAIP_SPECTRUM, NAIP_GRID,
                                       HARDCODED_NAIP_LIST).download_naips()

    road_labels, naip_tiles, waymap, way_bitmap_npy = random_training_data(
        raster_data_paths, args.extract_type, args.band_list, args.tile_size)
    equal_count_way_list, equal_count_tile_list = equalize_data(road_labels, naip_tiles,
                                                                args.save_clippings)
    test_labels, training_labels, test_images, training_images = split_train_test(
        equal_count_tile_list, equal_count_way_list)
    label_types = waymap.extracter.types

    print("SAVING DATA: pickling and saving to disk")
    t0 = time.time()
    cache_path = '/data/cache/'
    try:
        os.mkdir(cache_path)
    except:
        pass
    with open(cache_path + 'training_images.pickle', 'w') as outfile:
        pickle.dump(training_images, outfile)
    with open(cache_path + 'training_labels.pickle', 'w') as outfile:
        pickle.dump(training_labels, outfile)
    with open(cache_path + 'test_images.pickle', 'w') as outfile:
        pickle.dump(test_images, outfile)
    with open(cache_path + 'test_labels.pickle', 'w') as outfile:
        pickle.dump(test_labels, outfile)
    with open(cache_path + 'label_types.json', 'w') as outfile:
        json.dump(label_types, outfile)
    with open(cache_path + 'raster_data_paths.json', 'w') as outfile:
        json.dump(raster_data_paths, outfile)
    print("SAVE DONE: time to pickle/json and save test data to disk {0:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
