#!/usr/bin/env python

import argparse
import os
import pickle
import time

from src.download_naips import NAIPDownloader
from src.create_training_data import (NUMBER_OF_NAIPS, RANDOMIZE_NAIPS, NAIP_STATE, NAIP_RESOLUTION,
                                      NAIP_YEAR, NAIP_SPECTRUM, NAIP_GRID, HARDCODED_NAIP_LIST,
                                      random_training_data, equalize_data, split_train_test)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile_size",
                        default='64',
                        help="tile the NAIP and training data into NxN tiles with this dimension")
    parser.add_argument("--bands", default='0001', help="defaults to 0001 for just IR active")
    parser.add_argument("--extract_type", default='highway', help="highway or tennis")
    parser.add_argument("--cache_way_bmp",
                        default=True,
                        help="disable this to create way bitmaps each run")
    parser.add_argument("--clear_way_bmp_cache",
                        default=False,
                        help="enable this to bust the ay_bmp_cache from previous runs")
    parser.add_argument("--save_clippings",
                        default=False,
                        help="save the training data tiles to /data/naip")


def main():
    parser = create_parser()
    args = parser.parse_args()

    bands_string = args.bands
    band_list = []
    for char in bands_string:
        band_list.append(int(char))

    raster_data_paths = NAIPDownloader(NUMBER_OF_NAIPS, RANDOMIZE_NAIPS, NAIP_STATE, NAIP_YEAR,
                                       NAIP_RESOLUTION, NAIP_SPECTRUM, NAIP_GRID,
                                       HARDCODED_NAIP_LIST).download_naips()

    tile_size = int(args.tile_size)
    road_labels, naip_tiles, waymap, way_bitmap_npy = random_training_data(
        raster_data_paths, args.cache_way_bmp, args.clear_way_bmp_cache, args.extract_type,
        band_list, tile_size)
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
    with open(cache_path + 'label_types.pickle', 'w') as outfile:
        pickle.dump(label_types, outfile)
    with open(cache_path + 'raster_data_paths.pickle', 'w') as outfile:
        pickle.dump(raster_data_paths, outfile)
    with open(cache_path + 'way_bitmap_npy.pickle', 'w') as outfile:
        pickle.dump(way_bitmap_npy, outfile)
    print("SAVE DONE: time to pickle and save test data to disk {0:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
