import argparse
import numpy, os, sys, time
from random import shuffle
from osgeo import gdal
from PIL import Image
from pyproj import Proj, transform

from download_labels import WayMap, download_and_extract
from download_naips import NAIPDownloader
from geo_util import latLonToPixel, pixelToLatLng
import label_chunks_cnn
import label_chunks_cnn_cifar
from config_data import *

def read_naip(file_path, bands_to_use):
  '''
      read a NAIP from disk
      bands_to_use is an array of 4 Booleans, in whether to use each band (R, G, B, and IR)
      from http://www.machinalis.com/blog/python-for-geospatial-data-processing/
  '''
  raster_dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
  coord = pixelToLatLng(raster_dataset, 0, 0)
  proj = raster_dataset.GetProjectionRef()
  
  bands_data = []
  # 4 bands of raster data, RGB and IR
  index = 0
  for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    if bands_to_use[index] == 1:
      bands_data.append(band.ReadAsArray())
    index += 1
  bands_data = numpy.dstack(bands_data)
  
  return raster_dataset, bands_data

def tile_naip(raster_data_path, raster_dataset, bands_data, bands_to_use, tile_size):
  '''
     cut a 4-band raster image into tiles,
     tiles are cubes - up to 4 bands, and N height x N width based on tile_size settings
  '''
  on_band_count = 0
  for b in bands_to_use:
    if b == 1:
      on_band_count += 1

  rows, cols, n_bands = bands_data.shape
  print("OPENED NAIP with {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
  print("GEO-BOUNDS for image chunk is {}".format(bounds_for_naip(raster_dataset, rows, cols)))

  all_tiled_data = []

  for col in range(0, cols, tile_size):
    for row in range(0, rows, tile_size):
      if row+tile_size < rows and col+tile_size < cols:
        new_tile = bands_data[row:row+tile_size, col:col+tile_size,0:on_band_count]
        all_tiled_data.append((new_tile,(col, row),raster_data_path))
 
  return all_tiled_data

def way_bitmap_for_naip(ways, raster_data_path, raster_dataset, rows, cols, cache_way_bmp=False, clear_way_bmp_cache=False):
  '''
    generate a matrix of size rows x cols, initialized to all zeroes,
    but set to 1 for any pixel where an OSM way runs over
  '''
  cache_filename = raster_data_path + '-ways.bitmap.npy'

  if clear_way_bmp_cache:
    try:
      os.path.remove(cache_filename)
      print "DELETED: previously cached way bitmap"
      return arr
    except:
      pass
      # print "WARNING: no previously cached way bitmap to delete"
  else:    
    try:
      if cache_way_bmp:
        arr = numpy.load(cache_filename)
        print "CACHED: read label data from disk"
        return arr
    except:
      pass
      # print "ERROR reading bitmap cache from disk: {}".format(cache_filename)

  way_bitmap = numpy.zeros([rows, cols], dtype=numpy.int)
  bounds = bounds_for_naip(raster_dataset, rows, cols)
  ways_on_naip = []

  t0 = time.time()
  print("FINDING WAYS on NAIP..."),
  for way in ways:
    for point_tuple in way['linestring']:
      if bounds_contains_point(bounds, point_tuple):
        ways_on_naip.append(way)
        break
  print(" {0:.1f}s".format(time.time()-t0))
  print("EXTRACTED {} highways in NAIP bounds, of {} ways".format(len(ways_on_naip), len(ways)))

  print "MAKING BITMAP for way presence...",
  t0 = time.time()
  for w in ways_on_naip:
    for x in range(len(w['linestring'])-1):
      current_point = w['linestring'][x]
      next_point = w['linestring'][x+1]
      if not bounds_contains_point(bounds, current_point) or \
         not bounds_contains_point(bounds, next_point):
        continue
      current_pix = latLonToPixel(raster_dataset, current_point)
      next_pix = latLonToPixel(raster_dataset, next_point)
      add_pixels_between(current_pix, next_pix, cols, rows, way_bitmap)
  print(" {0:.1f}s".format(time.time()-t0))

  if cache_way_bmp and not os.path.exists(cache_filename):
    print "CACHING {}...", cache_filename,
    t0 = time.time()
    numpy.save(cache_filename, way_bitmap)
    print(" {0:.1f}s".format(time.time()-t0))

  return way_bitmap

def bounds_for_naip(raster_dataset, rows, cols):
  '''
      clip the NAIP to 0 to cols, 0 to rows
  '''
  left_x, right_x, top_y, bottom_y = 0, cols, 0, rows
  sw = pixelToLatLng(raster_dataset, left_x, bottom_y)
  ne = pixelToLatLng(raster_dataset, right_x, top_y)
  return {'sw': sw, 'ne': ne}

def add_pixels_between(start_pixel, end_pixel, cols, rows, way_bitmap):
  '''
      add the pixels between the start and end to way_bitmap,
      maybe thickened based on config
  '''
  if end_pixel[0] - start_pixel[0] == 0:
    for y in range(min(end_pixel[1], start_pixel[1]),
                   max(end_pixel[1], start_pixel[1])):
      safe_add_pixel(end_pixel[0], y, way_bitmap)
      # if configged, fatten lines
      for x in range(1,PIXELS_BESIDE_WAYS+1):
        safe_add_pixel(end_pixel[0]-x, y, way_bitmap)
        safe_add_pixel(end_pixel[0]+x, y, way_bitmap)
    return

  slope = (end_pixel[1] - start_pixel[1])/float(end_pixel[0] - start_pixel[0])
  offset = end_pixel[1] - slope*end_pixel[0]

  i = 0
  while i < cols:
    floatx = start_pixel[0] + (end_pixel[0] - start_pixel[0]) * i / float(cols)
    p = (int(floatx),int(offset + slope * floatx))
    safe_add_pixel(p[0],p[1], way_bitmap)
    i += 1
    # if configged, fatten lines
    for x in range(1, PIXELS_BESIDE_WAYS+1):
      safe_add_pixel(p[0], p[1]-x, way_bitmap)
      safe_add_pixel(p[0], p[1]+x, way_bitmap)
      safe_add_pixel(p[0]-x, p[1], way_bitmap)
      safe_add_pixel(p[0]+x, p[1], way_bitmap)

def safe_add_pixel(x, y, way_bitmap):
  '''
     turn on a pixel in way_bitmap if its in bounds
  '''
  if x < 0 or y < 0 or x >= len(way_bitmap[0]) or y >= len(way_bitmap):
    return
  way_bitmap[y][x] = 1

def bounds_contains_point(bounds, point_tuple):
  '''
     returns True if the bounds geographically contains the point_tuple
  '''
  if point_tuple[0] > bounds['ne'][0]:
    return False
  if point_tuple[0] < bounds['sw'][0]:
    return False
  if point_tuple[1] > bounds['ne'][1]:
    return False
  if point_tuple[1] < bounds['sw'][1]:
    return False
  return True

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
    if has_ways_in_center(label[0]):
      onehot_labels.append([0,1])
      on_count += 1
    elif not has_ways(label[0]):
      onehot_labels.append([1,0])
      off_count += 1

  print "ONE-HOT labels: {} on, {} off ({:.1%} on)".format(on_count, off_count, on_count/float(len(labels)))
  return onehot_labels


def has_ways(tile):
  '''  
     returns true if some pixels on the NxN tile are set to 1
  
  '''
  road_pixel_count = 0
  for x in range(0, len(tile)):
    for y in range(0, len(tile[x])):
      pixel_value = tile[x][y]
      if pixel_value != 0:
        road_pixel_count += 1
  if road_pixel_count >= len(tile)*PERCENT_OF_TILE_HEIGHT_TO_ACTIVATE:
    return True
  return False


def has_ways_in_center(tile):
  center_pixel_count = 0
  center_x = len(tile)/2
  center_y = len(tile[0])/2

  for x in range(0, len(tile)):
    for y in range(0, len(tile[x])):
      pixel_value = tile[x][y]
      if pixel_value != 0:
        if x >= center_x -1 and x <= center_x + 1:
          if y >= center_y -1 and y <= center_y + 1:
            center_pixel_count += 1
  if center_pixel_count >= 4:
    return True
  return False

def has_no_ways_in_fatter_center(tile):
  center_pixel_count = 0
  center_x = len(tile)/2
  center_y = len(tile[0])/2

  for x in range(0, len(tile)):
    for y in range(0, len(tile[x])):
      pixel_value = tile[x][y]
      if pixel_value != 0:
        if x >= center_x -5 and x <= center_x + 5:
          if y >= center_y -5 and y <= center_y + 5:
            center_pixel_count += 1
  if center_pixel_count <= 4:
    return True
  return False


def shuffle_in_unison(a, b):
   '''
       http://stackoverflow.com/questions/11765061/better-way-to-shuffle-two-related-lists
   '''
   a_shuf = []
   b_shuf = []
   index_shuf = range(len(a))
   shuffle(index_shuf)
   for i in index_shuf:
       a_shuf.append(a[i])
       b_shuf.append(b[i])
   return a_shuf, b_shuf

def create_training_data(cache_way_bmp, 
                         clear_way_bmp_cache, 
                         render_results, 
                         extract_type, 
                         band_list, 
                         tile_size, 
                         save_clippings):  
  raster_data_paths = NAIPDownloader(NUMBER_OF_NAIPS,
                                     RANDOMIZE_NAIPS,
                                     NAIP_STATE,
                                     NAIP_YEAR,
                                     NAIP_RESOLUTION,
                                     NAIP_SPECTRUM,
                                     NAIP_GRID,
                                     HARDCODED_NAIP_LIST).download_naips()  
  road_labels, naip_tiles, waymap, way_bitmap_npy = random_training_data(raster_data_paths, cache_way_bmp, clear_way_bmp_cache, extract_type, band_list, tile_size)
  equal_count_way_list, equal_count_tile_list = equalize_data(road_labels, naip_tiles, save_clippings)
  test_labels, training_labels, test_images, training_images = split_train_test(equal_count_tile_list,equal_count_way_list)
  return training_images, training_labels, test_images, test_labels, waymap

def run_analysis(training_images, training_labels, test_images, test_labels,  
                 waymap,
                 render_results, 
                 model, 
                 band_list, 
                 training_batches, 
                 batch_size, 
                 tile_size):  
  predictions = analyze(test_labels, training_labels, test_images, training_images, waymap, model, band_list, training_batches, batch_size, tile_size)
  if render_results:
    render_results_as_images(raster_data_paths, training_labels, test_labels, predictions, way_bitmap_npy, band_list, tile_size)

def random_training_data(raster_data_paths, cache_way_bmp, clear_way_bmp_cache, extract_type, band_list, tile_size):
  road_labels = []
  naip_tiles = []

  # tile images and labels  
  waymap = download_and_extract(PBF_FILE_URLS, extract_type)
  way_bitmap_npy = {}

  for raster_data_path in raster_data_paths:
    raster_dataset, bands_data = read_naip(raster_data_path, band_list)
    rows = bands_data.shape[0]
    cols = bands_data.shape[1]
  
    way_bitmap_npy[raster_data_path] = numpy.asarray(way_bitmap_for_naip(waymap.extracter.ways, raster_data_path, raster_dataset, rows, cols, cache_way_bmp, clear_way_bmp_cache))  

    left_x, right_x, top_y, bottom_y = 0, cols, 0, rows
    for row in range(top_y, bottom_y, tile_size):
      for col in range(left_x, right_x, tile_size):
        if row+tile_size < bottom_y and col+tile_size < right_x:
          new_tile = way_bitmap_npy[raster_data_path][row:row+tile_size, col:col+tile_size]
          road_labels.append((new_tile,(col, row),raster_data_path))
        
    for tile in tile_naip(raster_data_path, raster_dataset, bands_data, band_list, tile_size):
      naip_tiles.append(tile)

  assert len(road_labels) == len(naip_tiles)

  road_labels, naip_tiles = shuffle_in_unison(road_labels, naip_tiles)
  return road_labels, naip_tiles, waymap, way_bitmap_npy

def equalize_data(road_labels, naip_tiles, save_clippings):
  wayless_indices = []
  way_indices = []
  for x in range(len(road_labels)):
    tile = road_labels[x][0]
    if has_ways_in_center(tile):
      way_indices.append(x)
    elif not has_ways(tile):
      wayless_indices.append(x)

  count_wayless = len(wayless_indices)
  count_withways = len(way_indices)

  equal_count_way_list = []
  equal_count_tile_list = []
  for x in range(min(count_wayless, count_withways)):
    way_index = way_indices[x]
    wayless_index = wayless_indices[x]
    equal_count_way_list.append(road_labels[way_index])
    equal_count_way_list.append(road_labels[wayless_index])
    equal_count_tile_list.append(naip_tiles[way_index])
    if save_clippings:
      save_image_clipping(naip_tiles[way_index], 'ON')
    equal_count_tile_list.append(naip_tiles[wayless_index])
    if save_clippings:
      save_image_clipping(naip_tiles[wayless_index], 'OFF')
  return equal_count_way_list, equal_count_tile_list

def save_image_clipping(tile, status):
  # (new_tile,(col, row),raster_data_path)
  rgbir_matrix = tile[0]
  img = numpy.empty([64,64])
  for x in range(len(rgbir_matrix)):
    for y in range(len(rgbir_matrix[x])):
      img[x][y] = rgbir_matrix[x][y][0]
  im = Image.merge('RGB',(Image.fromarray(img).convert('L'),Image.fromarray(img).convert('L'),Image.fromarray(img).convert('L')))
  outfile_path = tile[2] + '-' + status + '-' + str(tile[1][0]) + ',' + str(tile[1][1]) + '-' + '.jpg'
  im.save(outfile_path, "JPEG")

def split_train_test(equal_count_tile_list,equal_count_way_list):
  test_labels = []
  training_labels = []
  test_images = []
  training_images = []

  for x in range(0, len(equal_count_way_list)):
    if PERCENT_FOR_TRAINING_DATA > float(x)/len(equal_count_tile_list):
      training_images.append(equal_count_tile_list[x])
      training_labels.append(equal_count_way_list[x])
    else:
      test_images.append(equal_count_tile_list[x])
      test_labels.append(equal_count_way_list[x])
  return test_labels, training_labels, test_images, training_images

def analyze(test_labels, training_labels, test_images, training_images, waymap, model, band_list, training_batches, batch_size, tile_size):
  ''' 
      package data for tensorflow and analyze
  '''
  print_data_dimensions(training_labels, band_list)
  onehot_training_labels, \
  onehot_test_labels = format_as_onehot_arrays(waymap.extracter.types, training_labels, test_labels)
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
  parser.add_argument("--save_clippings", default=False, help="save the training data tiles to /data/naip")
  parser.add_argument("--tile_size", default='64', help="tile the NAIP and training data into NxN tiles with this dimension")
  # the number of batches to train the neural net
  # @lacker recommends 3-5K for statistical significance, as rule of thumb
  # can achieve 90+% accuracy with 5000 so far
  # 100 is just so everything runs fast-ish and prints output, for a dry run
  parser.add_argument("--training_batches", default='100', help="set this to more like 5000 to make analysis work")
  parser.add_argument("--batch_size", default='96', help="around 100 is a good choice, defaults to 96 because cifar10 does")
  parser.add_argument("--bands", default='1111', help="defaults to 1111 for R+G+B+IR active")
  parser.add_argument("--extract_type", default='highway', help="highway or tennis")
  parser.add_argument("--cache_way_bmp", default=True, help="disable this to create way bitmaps each run")
  parser.add_argument("--clear_way_bmp_cache", default=False, help="enable this to bust the ay_bmp_cache from previous runs")
  parser.add_argument("--render_results", default=True, help="disable to not print data/predictions to JPEG")
  parser.add_argument("--model", default='cifar10', help="mnist or cifar10")
  args = parser.parse_args()
  bands_string = args.bands
  band_list = []
  for char in bands_string:
    band_list.append(int(char))

  training_images, training_labels, test_images, test_labels, waymap = \
      create_training_data(cache_way_bmp, 
                           clear_way_bmp_cache, 
                           extract_type=args.extract_type, 
                           band_list=band_list, 
                           tile_size=int(args.tile_size), 
                           save_clippings=args.save_clippings)  

  run_analysis(training_images, training_labels, test_images, test_labels, 
               waymap,
               render_results=args.render_results, 
               model=args.model, 
               band_list=band_list, 
               training_batches=args.training_batches, 
               batch_size=int(args.batch_size), 
               tile_size=int(args.tile_size)  

