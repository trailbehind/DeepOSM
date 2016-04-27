import argparse
import numpy, os, sys, time
from random import shuffle
from osgeo import gdal
from PIL import Image
from pyproj import Proj, transform

from download_labels import WayMap, download_and_extract
from download_naips import NAIPDownloader
from geo_util import latLonToPixel, pixelToLatLng
from label_chunks_cnn import train_neural_net

# tile the NAIP and training data into NxN tiles with this dimension
TILE_SIZE = 64

# the remainder is allocated as test data
PERCENT_FOR_TRAINING_DATA = .95

# the bands to use from the NAIP for analysis (R G B IR)
BANDS_TO_USE = [0,0,0,1]

def read_naip(file_path, bands_to_use):
  '''
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

def tile_naip(raster_data_path, raster_dataset, bands_data, bands_to_use):
  '''
     cut a 4-band raster image into tiles,
     tiles are cubes - up to 4 bands, and N height x N width based on TILE_SIZE settings
  '''
  on_band_count = 0
  for b in bands_to_use:
    if b == 1:
      on_band_count += 1

  rows, cols, n_bands = bands_data.shape
  print("OPENED NAIP with {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
  print("GEO-BOUNDS for image chunk is {}".format(bounds_for_naip(raster_dataset, rows, cols)))

  all_tiled_data = []

  left_x, right_x, top_y, bottom_y = 0, cols, 0, rows

  for col in range(left_x, right_x, TILE_SIZE):
    for row in range(top_y, bottom_y, TILE_SIZE):
      if row+TILE_SIZE < bottom_y and col+TILE_SIZE < right_x:
        new_tile = bands_data[row:row+TILE_SIZE, col:col+TILE_SIZE,0:on_band_count]
        all_tiled_data.append((new_tile,(col, row),raster_data_path))
 
  return all_tiled_data

def way_bitmap_for_naip(ways, raster_data_path, raster_dataset, rows, cols, use_pbf_cache=True):
  '''
    generate a matrix of size rows x cols, initialized to all zeroes,
    but set to 1 for any pixel where an OSM way runs over
  '''
  cache_filename = raster_data_path + '-ways.bitmap'
  try:
    if use_pbf_cache:
      arr = numpy.load(cache_filename)
      print "CACHED: read label data from disk"
      return arr
    else:
      print "CREATING LABELS from PBF file"
  except:
    print "CREATING LABELS from PBF file"

  way_bitmap = empty_tile_matrix(rows, cols)
  bounds = bounds_for_naip(raster_dataset, rows, cols)
  ways_on_naip = []
  for way in ways:
    for point_tuple in way['linestring']:
      if bounds_contains_point(bounds, point_tuple):
        ways_on_naip.append(way)
        break
  print("EXTRACTED {} highways that overlap the NAIP, out of {} ways in the PBF".format(len(ways_on_naip), len(ways)))

  for w in ways_on_naip:
    for x in range(len(w['linestring'])-1):
      current_point = w['linestring'][x]
      next_point = w['linestring'][x+1]
      if not bounds_contains_point(bounds, current_point) or \
         not bounds_contains_point(bounds, next_point):
        continue
      current_pix = latLonToPixel(raster_dataset, current_point)
      next_pix = latLonToPixel(raster_dataset, next_point)
      pixel_line = pixels_between(current_pix, next_pix, cols)
      for p in pixel_line:
        if p[0] < 0 or p[1] < 0 or p[0] >= cols or p[1] >= rows:
          continue
        else:
          way_bitmap[p[1]][p[0]] = w['highway_type']

  print "CACHING way_bitmap numpy array to", cache_filename
  numpy.save(cache_filename, way_bitmap)

  return way_bitmap

def empty_tile_matrix(rows, cols):
  '''
      initialize the array to all zeroes
  '''
  tile_matrix = []
  for x in range(0,rows):
    tile_matrix.append([])
    for y in range(0,cols):
      tile_matrix[x].append(0)
  return tile_matrix

def bounds_for_naip(raster_dataset, rows, cols):
  '''
      clip the NAIP to 0 to cols, 0 to rows
  '''
  left_x, right_x, top_y, bottom_y = 0, cols, 0, rows
  sw = pixelToLatLng(raster_dataset, left_x, bottom_y)
  ne = pixelToLatLng(raster_dataset, right_x, top_y)
  return {'sw': sw, 'ne': ne}

def pixels_between(start_pixel, end_pixel, cols):
  '''
      returns a list of pixel tuples between current and next, inclusive
  '''
  pixels = []
  if end_pixel[0] - start_pixel[0] == 0:
    for y in range(min(end_pixel[1], start_pixel[1]),
                   max(end_pixel[1], start_pixel[1])):
      p = []
      p.append(end_pixel[0])
      p.append(y)
      pixels.append(p)
      pixels.append([p[0]-1, p[1]])
      pixels.append([p[0]+1, p[1]])
    return pixels

  slope = (end_pixel[1] - start_pixel[1])/float(end_pixel[0] - start_pixel[0])
  offset = end_pixel[1] - slope*end_pixel[0]

  i = 0
  while i < cols:
    p = []
    floatx = start_pixel[0] + (end_pixel[0] - start_pixel[0]) * i / float(cols)
    p.append(int(floatx))
    p.append(int(offset + slope * floatx))
    i += 1
    if not p in pixels:
      pixels.append(p)

    # make lines 3px thick
    if slope == 0:
      top_p = [p[0], p[1]-1]
      if not top_p in pixels:
        pixels.append(top_p)
      bottom_p = [p[0], p[1]+1]
      if not bottom_p in pixels:
        pixels.append(bottom_p)

    else:
      left_p = [p[0]-1, p[1]]
      if not left_p in pixels:
        pixels.append(left_p)
      right_p = [p[0]+1, p[1]]
      if not right_p in pixels:
        pixels.append(right_p)


  return pixels

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

    types_hot = []
    for highway_type in types:
      types_hot.append(0)
  '''
  
  print "CREATING TEST one-hot labels"
  onehot_test_labels = onehot_for_labels(test_labels)
  print "CREATING TRAINING one-hot labels"
  onehot_training_labels = onehot_for_labels(training_labels)

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
    else:
      onehot_labels.append([1,0])
      off_count += 1

  print "ONE-HOT labels: {} on, {} off ({:.1%} on)".format(on_count, off_count, on_count/float(len(labels)))
  return onehot_labels

def has_ways(tile):
  '''
     returns true if any pixel on the NxN tile is set to 1
  '''
  road_pixel_count = 0
  for x in range(0, len(tile)):
    for y in range(0, len(tile[x])):
      pixel_value = tile[x][y]
      if pixel_value != '0':
        road_pixel_count += 1
  if road_pixel_count >= len(tile)*.25:
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

def run_analysis(use_pbf_cache=False, render_results=True):  
  raster_data_paths = NAIPDownloader().download_naips()  
  road_labels, naip_tiles, waymap, way_bitmap_npy = random_training_data(raster_data_paths, use_pbf_cache)
  equal_count_way_list, equal_count_tile_list = equalize_data(road_labels, naip_tiles)
  test_labels, training_labels, test_images, training_images = split_train_test(equal_count_tile_list,equal_count_way_list)
  predictions = analyze(test_labels, training_labels, test_images, training_images, waymap)
  render_results_as_images(raster_data_paths, training_labels, test_labels, predictions, way_bitmap_npy)

def random_training_data(raster_data_paths, use_pbf_cache):
  road_labels = []
  naip_tiles = []

  # tile images and labels  
  waymap = download_and_extract()
  way_bitmap_npy = {}

  for raster_data_path in raster_data_paths:
    raster_dataset, bands_data = read_naip(raster_data_path, BANDS_TO_USE)
    rows = bands_data.shape[0]
    cols = bands_data.shape[1]
  
    way_bitmap_npy[raster_data_path] = numpy.asarray(way_bitmap_for_naip(waymap.extracter.ways, raster_data_path, raster_dataset, rows, cols, use_pbf_cache))  

    left_x, right_x, top_y, bottom_y = 0, cols, 0, rows
    for row in range(top_y, bottom_y, TILE_SIZE):
      for col in range(left_x, right_x, TILE_SIZE):
        if row+TILE_SIZE < bottom_y and col+TILE_SIZE < right_x:
          new_tile = way_bitmap_npy[raster_data_path][row:row+TILE_SIZE, col:col+TILE_SIZE]
          road_labels.append((new_tile,(col, row),raster_data_path))
        
    for tile in tile_naip(raster_data_path, raster_dataset, bands_data, BANDS_TO_USE):
      naip_tiles.append(tile)

  assert len(road_labels) == len(naip_tiles)

  road_labels, naip_tiles = shuffle_in_unison(road_labels, naip_tiles)
  return road_labels, naip_tiles, waymap, way_bitmap_npy

def equalize_data(road_labels, naip_tiles):
  wayless_indices = []
  way_indices = []
  for x in range(len(road_labels)):
    tile = road_labels[x][0]
    if has_ways(tile):
      way_indices.append(x)
    else:
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
    equal_count_tile_list.append(naip_tiles[wayless_index])
  return equal_count_way_list, equal_count_tile_list

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

def analyze(test_labels, training_labels, test_images, training_images, waymap):
  ''' 
      package data for tensorflow and analyze
  '''
  print_data_dimensions(training_labels)
  onehot_training_labels, \
  onehot_test_labels = format_as_onehot_arrays(waymap.extracter.types, training_labels, test_labels)
  npy_training_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in training_images])
  
  npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
  npy_training_labels = numpy.asarray(onehot_training_labels)
  npy_test_labels = numpy.asarray(onehot_test_labels)

  # train and test the neural net
  predictions = train_neural_net(BANDS_TO_USE, TILE_SIZE,
                   npy_training_images, 
                   npy_training_labels, 
                   npy_test_images, 
                   npy_test_labels)
  return predictions

def print_data_dimensions(training_labels):
  tiles = len(training_labels)
  h = len(training_labels[0][0])
  w = len(training_labels[0][0][0])
  bands = len(training_labels[0][0][0][0])
  print("TRAINING/TEST DATA: shaped the tiff data to {} tiles sized {} x {} with {} bands".format(tiles*2, h, w, bands))

def render_results_as_images(raster_data_paths, training_labels, test_labels, predictions, way_bitmap_npy):
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
    test_labels_by_naip[label[2]].append(predictions[index])
    index += 1

  index = 0
  for label in training_labels:
    training_labels_by_naip[label[2]].append(training_labels[index])
    index += 1

  for raster_data_path in raster_data_paths:
    if render_results:
      render_results_as_image(raster_data_path, 
                              way_bitmap_npy[raster_data_path], 
                              training_labels_by_naip[raster_data_path], 
                              test_labels_by_naip[raster_data_path], 
                              predictions=predictions_by_naip[raster_data_path])


def render_results_as_image(raster_data_path, way_bitmap, training_labels, test_labels, predictions=None):
  '''
      save the source TIFF as a JPEG, with labels and data overlaid
  '''
  timestr = time.strftime("%Y%m%d-%H%M%S")
  outfile = os.path.splitext(raster_data_path)[0] + '-' + timestr + ".png"
  im = Image.open(raster_data_path)
  print "GENERATING PNG for %s" % raster_data_path
  rows = len(way_bitmap)
  cols = len(way_bitmap[0])

  # TIFF to JPEG bit from: from: http://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python
  t0 = time.time()
  r, g, b, ir = im.split()
  for x in range(cols):
    for y in range(rows):
      ir.putpixel((x, y),(0))    
  im = Image.merge("RGBA", (r, g, b, r))
  t1 = time.time()
  print "{} elapsed to FLATTEN 4 BAND TIFF TO PNG".format(t1-t0)

  t0 = time.time()
  shade_labels(im, test_labels, predictions)
  t1 = time.time()
  print "{} elapsed to SHADE PREDICTIONS ON PNG".format(t1-t0)

  t0 = time.time()
  # show raw data that spawned the labels
  for row in range(0, rows):
    for col in range(0, cols):
      if way_bitmap[row][col] == 'primary':
        im.putpixel((col, row), (255,0,0, 255))
      elif way_bitmap[row][col] == 'trunk':
        im.putpixel((col, row), (0,255,0, 255))
      elif way_bitmap[row][col] != '0':
        # secondary and tertiary
        im.putpixel((col, row), (0,0,255, 255))
  t1 = time.time()
  print "{} elapsed to DRAW WAYS ON PNG".format(t1-t0)

  im.save(outfile, "PNG")

def shade_labels(image, labels, predictions):
  '''
      visualize predicted ON labels as blue, OFF as green
  '''
  label_index = 0
  for label in labels:
    start_x = label[1][0]
    start_y = label[1][1]
    for x in range(start_x, start_x+TILE_SIZE):
      for y in range(start_y, start_y+TILE_SIZE):
        r, g, b, a = image.getpixel((x, y))
        if predictions[label_index] == 1:
          # shade ON predictions blue
          image.putpixel((x, y), (r, g, 255, 255))
        else:
          # shade OFF predictions green
          image.putpixel((x, y), (r, 255, b, 255))
    label_index += 1

parser = argparse.ArgumentParser()
parser.add_argument("--use_pbf_cache", default=False, help="enable this to not reparse PBF")
parser.add_argument("--render_results", default=True, help="enable this to print data/predictions to JPEG")
args = parser.parse_args()
render_results = False
if args.render_results:
  render_results = True
if args.use_pbf_cache:
  run_analysis(use_pbf_cache=True, render_results=render_results)
else:
  run_analysis(use_pbf_cache=False, render_results=render_results)
