from __future__ import print_function

import numpy, os, sys, time
from PIL import Image
import label_chunks_cnn
import label_chunks_cnn_cifar
from config_data import *
from create_training_data import has_ways


def analyze(test_labels, training_labels, test_images, training_images, label_types, model, band_list, training_batches, batch_size, tile_size):
  '''
      package data for tensorflow and analyze
  '''
  print_data_dimensions(training_labels, band_list)
  onehot_training_labels, \
  onehot_test_labels = format_as_onehot_arrays(label_types, training_labels, test_labels)
  npy_training_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in training_images])

  npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
  npy_training_labels = numpy.asarray(onehot_training_labels)
  npy_test_labels = numpy.asarray(onehot_test_labels)

  # train and test the neural net
  predictions = None
  if model == 'mnist':
    predictions = label_chunks_cnn.train_neural_net(band_list,
                                                 tile_size,
                                                 npy_training_images,
                                                 npy_training_labels,
                                                 npy_test_images,
                                                 npy_test_labels,
                                                 CONVOLUTION_PATCH_SIZE,
                                                 training_batches,
                                                 batch_size)
  elif model == 'cifar10':
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

def format_as_onehot_arrays(types, training_labels, test_labels):
  '''
     each label gets converted from an NxN tile with way bits flipped,
     into a one hot array of whether the tile contains ways (i.e. [0,1] or [1,0] for each)
  '''
  print("CREATING ONE-HOT LABELS...")
  t0 = time.time()
  print("CREATING TEST one-hot labels")
  onehot_test_labels = onehot_for_labels(test_labels)
  print("CREATING TRAINING one-hot labels")
  onehot_training_labels = onehot_for_labels(training_labels)
  print("one-hotting took {0:.1f}s".format(time.time()-t0))

  return onehot_training_labels, onehot_test_labels

def onehot_for_labels(labels):
  '''
     returns a list of one-hot array labels, for a list of tiles
  '''
  on_count = 0
  off_count = 0

  onehot_labels = []
  for label in labels:
    if has_ways(label[0]):
      onehot_labels.append([0,1])
      on_count += 1
    elif not has_ways(label[0]):
      onehot_labels.append([1,0])
      off_count += 1

  print("ONE-HOT labels: {} on, {} off ({:.1%} on)".format(on_count, off_count, on_count/float(len(labels))))
  return onehot_labels

def print_data_dimensions(training_labels,band_list):
  tiles = len(training_labels)
  h = len(training_labels[0][0])
  w = len(training_labels[0][0][0])
  bands = sum(band_list)
  print("TRAINING/TEST DATA: shaped the tiff data to {} tiles sized {} x {} with {} bands".format(tiles*2, h, w, bands))

if __name__ == "__main__":
    print("Instead of running this file, use bin/run_analysis.py instead.", file=sys.stderr)
    sys.exit(1)
