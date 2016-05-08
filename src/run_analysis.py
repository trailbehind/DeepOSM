import argparse
import numpy, os, sys, time, pickle
from PIL import Image
import label_chunks_cnn
import label_chunks_cnn_cifar
from config_data import *
from create_training_data import has_ways, has_ways_in_center, has_no_ways_in_fatter_center


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
    print "ERROR, unknown model to use for analysis"
  return predictions

def format_as_onehot_arrays(types, training_labels, test_labels):
  '''
     each label gets converted from an NxN tile with way bits flipped,
     into a one hot array of whether the tile contains ways (i.e. [0,1] or [1,0] for each)
  '''
  print("CREATING ONE-HOT LABELS...")
  t0 = time.time()
  print "CREATING TEST one-hot labels"
  onehot_test_labels = onehot_for_labels(test_labels)
  print "CREATING TRAINING one-hot labels"
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

  print "ONE-HOT labels: {} on, {} off ({:.1%} on)".format(on_count, off_count, on_count/float(len(labels)))
  return onehot_labels

def print_data_dimensions(training_labels,band_list):
  tiles = len(training_labels)
  h = len(training_labels[0][0])
  w = len(training_labels[0][0][0])
  bands = sum(band_list)
  print("TRAINING/TEST DATA: shaped the tiff data to {} tiles sized {} x {} with {} bands".format(tiles*2, h, w, bands))

def render_results_as_images(raster_data_paths, training_labels, test_labels, predictions, way_bitmap_npy, band_list, tile_size):
  training_labels_by_naip = {}
  test_labels_by_naip = {}
  predictions_by_naip = {}
  for raster_data_path in raster_data_paths:
    predictions_by_naip[raster_data_path] = []
    test_labels_by_naip[raster_data_path] = []
    training_labels_by_naip[raster_data_path] = []

  index = 0
  for label in test_labels:
    predictions_by_naip[label[2]].append(predictions[index])
    test_labels_by_naip[label[2]].append(test_labels[index])
    index += 1

  index = 0
  for label in training_labels:
    training_labels_by_naip[label[2]].append(training_labels[index])
    index += 1

  for raster_data_path in raster_data_paths:
    render_results_as_image(raster_data_path, 
                            way_bitmap_npy[raster_data_path], 
                            training_labels_by_naip[raster_data_path], 
                            test_labels_by_naip[raster_data_path],
                            band_list, 
                            tile_size,
                            predictions=predictions_by_naip[raster_data_path])

def render_results_as_image(raster_data_path, way_bitmap, training_labels, test_labels, band_list, tile_size, predictions=None):
  '''
      save the source TIFF as a JPEG, with labels and data overlaid
  '''
  timestr = time.strftime("%Y%m%d-%H%M%S")
  outfile = os.path.splitext(raster_data_path)[0] + '-' + timestr + ".jpeg"
  # TIF to JPEG bit from: from: http://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python
  im = Image.open(raster_data_path)
  print "GENERATING JPEG for %s" % raster_data_path
  rows = len(way_bitmap)
  cols = len(way_bitmap[0])
  t0 = time.time()
  r, g, b, ir = im.split()
  # visualize single band analysis tinted for R-G-B, 
  # or grayscale for infrared band  
  if sum(band_list) == 1:
    if band_list[3] == 1:
      # visualize IR as grayscale
      im = Image.merge("RGB", (ir, ir, ir))
    else:
      # visualize single-color band analysis as a scale of that color
      zeros_band = Image.new('RGB', r.size).split()[0]
      if band_list[0] == 1:
        im = Image.merge("RGB", (r, zeros_band, zeros_band))
      elif band_list[1] == 1:
        im = Image.merge("RGB", (zeros_band, g, zeros_band))
      elif band_list[2] == 1:
        im = Image.merge("RGB", (zeros_band, zeros_band, b))
  else:
    # visualize multi-band analysis as RGB  
    im = Image.merge("RGB", (r, g, b))

  t1 = time.time()
  print "{0:.1f}s to FLATTEN the {1} analyzed bands of TIF to JPEG".format(t1-t0, sum(band_list))

  t0 = time.time()
  shade_labels(im, test_labels, predictions, tile_size)
  t1 = time.time()
  print "{0:.1f}s to SHADE PREDICTIONS on JPEG".format(t1-t0)

  t0 = time.time()
  # show raw data that spawned the labels
  for row in range(0, rows):
    for col in range(0, cols):
      if way_bitmap[row][col] != 0:
        im.putpixel((col, row), (255,0,0))
  t1 = time.time()
  print "{0:.1f}s to DRAW WAYS ON JPEG".format(t1-t0)

  im.save(outfile, "JPEG")

def shade_labels(image, labels, predictions, tile_size):
  '''
      visualize predicted ON labels as blue, OFF as green
  '''
  label_index = 0
  for label in labels:
    start_x = label[1][0]
    start_y = label[1][1]
    for x in range(start_x, start_x+tile_size):
      for y in range(start_y, start_y+tile_size):
        r, g, b = image.getpixel((x, y))
        if predictions[label_index] == 1:
          # shade ON predictions blue
          image.putpixel((x, y), (r, g, 255))
        else:
          # shade OFF predictions green
          image.putpixel((x, y), (r, 255, b))
    label_index += 1


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # the number of batches to train the neural net
  # @lacker recommends 3-5K for statistical significance, as rule of thumb
  # can achieve 90+% accuracy with 5000 so far
  # 100 is just so everything runs fast-ish and prints output, for a dry run
  parser.add_argument("--training_batches", default='6000', help="set this to ~5000 to make analysis work")
  parser.add_argument("--tile_size", default='64', help="tile the NAIP and training data into NxN tiles with this dimension")
  parser.add_argument("--batch_size", default='100', help="around 100 is a good choice, defaults to 96 because cifar10 does")
  parser.add_argument("--bands", default='0001', help="defaults to 0001 for just IR active")
  parser.add_argument("--extract_type", default='highway', help="highway or tennis")
  parser.add_argument("--render_results", default=True, help="disable to not print data/predictions to JPEG")
  parser.add_argument("--model", default='mnist', help="mnist or cifar10")
  args = parser.parse_args()
  bands_string = args.bands
  band_list = []
  for char in bands_string:
    band_list.append(int(char))

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
  print("DATA LOADED: time to unpickle test data {0:.1f}s".format(time.time()-t0))

  predictions = analyze(test_labels, 
                        training_labels, 
                        test_images, 
                        training_images, 
                        label_types, 
                        args.model, 
                        band_list, 
                        int(args.training_batches), 
                        int(args.batch_size), 
                        int(args.tile_size))
  
  if render_results:
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
                             band_list, 
                             int(args.tile_size))
