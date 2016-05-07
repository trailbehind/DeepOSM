#!/usr/bin/env python

import argparse
from src.run_analysis import run_analysis


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
    parser.add_argument("--extract-type", default='highway', choices=['highway', 'tennis'],
                        help="the type of feature to identify")
    parser.add_argument("--cache-way-bmp", action='store_true',
                        help="disable this to regenerate way bitmaps each run")
    parser.add_argument("--clear-way-bmp-cache", action='store_true',
                        help="enable this to bust the ay_bmp_cache from previous runs")
    parser.add_argument("--render-results", action='store_true',
                        help="disable to not print data/predictions to JPEG")
    parser.add_argument("--model", default='mnist', choices=['mnist', 'cifar10'],
                        help="the model to use")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    run_analysis(
        cache_way_bmp=args.cache_way_bmp,
        clear_way_bmp_cache=args.clear_way_bmp_cache,
        render_results=args.render_results,
        extract_type=args.extract_type,
        model=args.model,
        band_list=args.band_list,
        training_batches=args.training_batches,
        batch_size=args.batch_size,
        tile_size=args.tile_size)


if __name__ == "__main__":
    main()
