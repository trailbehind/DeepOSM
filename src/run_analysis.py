from __future__ import print_function

import sys

import numpy

import single_layer_network


def analyze(onehot_training_labels, onehot_test_labels, test_labels, training_labels, test_images,
            training_images, label_types, neural_net_type, band_list, tile_size, number_of_epochs):
    '''
      package data for tensorflow and analyze
    '''
    print_data_dimensions(training_labels, band_list)
    npy_training_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in training_images])

    npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
    npy_training_labels = numpy.asarray(onehot_training_labels)
    npy_test_labels = numpy.asarray(onehot_test_labels)

    # train and test the neural net
    predictions = single_layer_network.train(band_list, tile_size, npy_training_images,
                                             npy_training_labels, npy_test_images, npy_test_labels,
                                             number_of_epochs, neural_net_type)
    return predictions


def print_data_dimensions(training_labels, band_list):
    tiles = len(training_labels)
    h = len(training_labels[0][0])
    w = len(training_labels[0][0][0])
    bands = sum(band_list)
    print("TRAINING/TEST DATA: shaped the tiff data to {} tiles sized {} x {} with {} bands".format(
        tiles * 2, h, w, bands))


if __name__ == "__main__":
    print("Instead of running this file, use bin/run_analysis.py instead.", file=sys.stderr)
    sys.exit(1)
