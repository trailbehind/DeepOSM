from __future__ import print_function

import numpy, os, sys, time
from PIL import Image
import label_chunks_cnn
import label_chunks_cnn_cifar
from config_data import *
from create_training_data import has_ways, has_ways_in_center, has_no_ways_in_fatter_center


def analyze(onehot_training_labels, onehot_test_labels, test_labels, training_labels, test_images, training_images, label_types, model, band_list, training_batches, batch_size, tile_size):
  '''
      package data for tensorflow and analyze
  '''
  print_data_dimensions(training_labels, band_list)
  npy_training_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in training_images])

  npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
  npy_training_labels = numpy.asarray(onehot_training_labels)
  npy_test_labels = numpy.asarray(onehot_test_labels)

  # train and test the neural net
  predictions = None
  if model == '1conv':
      predictions = label_chunks_cnn_cifar.train_neural_net(
                                                 CONVOLUTION_PATCH_SIZE,
                                                 band_list,
                                                 tile_size,
                                                 npy_training_images,
                                                 npy_training_labels,
                                                 npy_test_images,
                                                 npy_test_labels,
                                                 training_batches,
                                                 batch_size)
  else:
    print("ERROR, unknown model to use for analysis")
  return predictions

def print_data_dimensions(training_labels,band_list):
  tiles = len(training_labels)
  h = len(training_labels[0][0])
  w = len(training_labels[0][0][0])
  bands = sum(band_list)
  print("TRAINING/TEST DATA: shaped the tiff data to {} tiles sized {} x {} with {} bands".format(tiles*2, h, w, bands))


if __name__ == "__main__":
    print("Instead of running this file, use bin/run_analysis.py instead.", file=sys.stderr)
    sys.exit(1)
