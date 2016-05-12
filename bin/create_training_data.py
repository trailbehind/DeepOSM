#!/usr/bin/env python

import argparse

from src.download_naips import NAIPDownloader
from src.create_training_data import (random_training_data, equalize_data, split_train_test, 
                                      format_as_onehot_arrays, dump_data_to_disk)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-size",
                        default=64,
                        type=int,
                        help="tile the NAIP and training data into NxN tiles with this dimension")
    parser.add_argument("--pixels-to-fatten-roads",
                        default=3,
                        type=int,
                        help="the number of pixels to fatten a road centerline (e.g. the default 3 makes roads 7px wide)")
    parser.add_argument("--percent-for-training-data",
                        default=.90,
                        type=float,
                        help="how much data to allocate for training, the remainder is left for test")
    parser.add_argument("--band-list",
                        default=[0, 0, 0, 1],
                        nargs=4,
                        type=int,
                        help="specify which bands to activate (R  G  B  IR)"
                        "--bands 0 0 0 1 (which activates only the IR band)")
    parser.add_argument("--naip-path",
                        default=['md', '2013', '1m', 'rgbir', '38077'],
                        nargs=5,
                        type=str,
                        help="values to create the S3 bucket path for some NAIPs"
                        "--bands md 2013 1m rgbir 38077 (defaults to some Maryland data)")
    parser.add_argument("--randomize-naips",
                        default=False,
                        type=bool,
                        help="set this to True if you don't want to get NAIPs in order from the bucket path")
    parser.add_argument("--number-of-naips",
                        default=5,
                        type=int,
                        help="set this to a value between 1 and 14 or so, 10 segfaults on a VirtualBox with 12GB, but runs on a Linux machine with 32GB")
    parser.add_argument("--label-data-files",
                        default=['http://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf',
                                 'http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf'],
                        type=str,
                        help="PBF files to extract road/feature label info from"
                        )
    parser.add_argument("--extract-type",
                        default='highway',
                        choices=['highway', 'tennis'],
                        help="the type of feature to identify")
    parser.add_argument("--save-clippings",
                        action='store_true',
                        help="save the training data tiles to /data/naip")
    return parser

NAIP_STATE, NAIP_YEAR, NAIP_RESOLUTION, NAIP_SPECTRUM, NAIP_GRID = args.naip_path

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    raster_data_paths = NAIPDownloader(args.number_of_naips, 
                                       args.randomize_naips, 
                                       NAIP_STATE, 
                                       NAIP_YEAR,
                                       NAIP_RESOLUTION, 
                                       NAIP_SPECTRUM, 
                                       NAIP_GRID,
                                       ).download_naips()
    road_labels, naip_tiles, waymap, way_bitmap_npy = random_training_data(
        raster_data_paths, args.extract_type, args.band_list, args.tile_size, args.pixels_to_fatten_roads, args.label_data_files)
    equal_count_way_list, equal_count_tile_list = equalize_data(road_labels, naip_tiles,
                                                                args.save_clippings)
    test_labels, training_labels, test_images, training_images = split_train_test(
        equal_count_tile_list, equal_count_way_list, args.percent_for_training_data)
    label_types = waymap.extracter.types
    onehot_training_labels, onehot_test_labels = format_as_onehot_arrays(label_types, training_labels, test_labels)
    dump_data_to_disk(raster_data_paths, training_images, training_labels, test_images, test_labels,
                      label_types, onehot_training_labels, onehot_test_labels)
                        

if __name__ == "__main__":
    main()
