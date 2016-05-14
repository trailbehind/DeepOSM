#!/usr/bin/env python

import argparse

from src.download_naips import NAIPDownloader
from src.create_training_data import (NUMBER_OF_NAIPS, RANDOMIZE_NAIPS, NAIP_STATE, NAIP_RESOLUTION,
                                      NAIP_YEAR, NAIP_SPECTRUM, NAIP_GRID, HARDCODED_NAIP_LIST,
                                      random_training_data, equalize_data, split_train_test, 
                                      format_as_onehot_arrays, dump_data_to_disk)


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
    parser.add_argument("--save-clippings",
                        action='store_true',
                        help="save the training data tiles to /data/naip")
    return parser
def main():
    parser = create_parser()
    args = parser.parse_args()
    
    raster_data_paths = NAIPDownloader(NUMBER_OF_NAIPS, RANDOMIZE_NAIPS, NAIP_STATE, NAIP_YEAR,
                                       NAIP_RESOLUTION, NAIP_SPECTRUM, NAIP_GRID,
                                       HARDCODED_NAIP_LIST).download_naips()
    road_labels, naip_tiles, waymap = random_training_data(
        raster_data_paths, args.extract_type, args.band_list, args.tile_size)
    equal_count_way_list, equal_count_tile_list = equalize_data(road_labels, naip_tiles,
                                                                args.save_clippings)
    test_labels, training_labels, test_images, training_images = split_train_test(
        equal_count_tile_list, equal_count_way_list)
    label_types = waymap.extracter.types
    onehot_training_labels, onehot_test_labels = format_as_onehot_arrays(label_types, training_labels, test_labels)
    dump_data_to_disk(raster_data_paths, training_images, training_labels, test_images, test_labels,
                      label_types, onehot_training_labels, onehot_test_labels)
                        

if __name__ == "__main__":
    main()
