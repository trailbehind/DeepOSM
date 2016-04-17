import numpy, os
from random import shuffle
from osgeo import gdal
from PIL import Image
from pyproj import Proj, transform
from extract_ways import WayMap, download_file
from download_naips import NAIPDownloader
from geo_util import latLonToPixel, pixelToLatLng
from label_chunks_cnn import train_neural_net

# tile the NAIP and training data into NxN tiles with this dimension
TILE_SIZE = 12

# the remainder is allocated as test data
PERCENT_FOR_TRAINING_DATA = .8

# big center chunk that avoids lack of data in Maryland for this PBF/NAIP combo
TOP_Y = 2500
BOTTOM_Y = 6500
LEFT_X = 500
RIGHT_X = 4000

'''
# small city chunk in middle
TOP_Y = 3500
BOTTOM_Y = 4500
LEFT_X = 2700
RIGHT_X = 3200
'''

GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR") # set in Dockerfile as env variable
DEFAULT_WAY_BITMAP_NPY_FILE = os.path.join(GEO_DATA_DIR, "way_bitmap.npy")

def read_naip(file_path):
  '''
      from http://www.machinalis.com/blog/python-for-geospatial-data-processing/
  '''
  raster_dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
  coord = pixelToLatLng(raster_dataset, 0, 0)
  proj = raster_dataset.GetProjectionRef()
  
  bands_data = []
  for b in range(1, raster_dataset.RasterCount+1):
    # just using the IR band for now
    # if b == 1:
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())
  bands_data = numpy.dstack(bands_data)
  
  training_images, test_images = tile_naip(raster_dataset, bands_data)
  
  return training_images, test_images, raster_dataset, bands_data

def tile_naip(raster_dataset, bands_data):
  rows, cols, n_bands = bands_data.shape
  print("OPENED NAIP with {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
  print("GEO-BOUNDS for image chunk is {}".format(bounds_for_naip(raster_dataset)))

  training_tiled_data = []
  test_tiled_data = []
  for col in range(LEFT_X, RIGHT_X-TILE_SIZE, TILE_SIZE):
    for row in range(TOP_Y, BOTTOM_Y-TILE_SIZE, TILE_SIZE):
      new_tile = bands_data[row:row+TILE_SIZE, col:col+TILE_SIZE,0:4]
      if row < (TOP_Y + (BOTTOM_Y-TOP_Y)*PERCENT_FOR_TRAINING_DATA):
        training_tiled_data.append((new_tile,(col, row)))
      else:
        test_tiled_data.append((new_tile,(col, row)))

  return training_tiled_data, test_tiled_data

def way_bitmap_for_naip(ways, raster_dataset, rows, cols):
  '''
    generate a matrix of size rows x cols, initialized to all zeroes,
    but set to 1 for any pixel where an OSM way runs over
  '''
  try:
    arr = numpy.load(DEFAULT_WAY_BITMAP_NPY_FILE)
    print "CACHED: read label data from disk"
    return arr
  except:
    print "CREATING LABELS"

  way_bitmap = empty_tile_matrix(rows, cols)
  bounds = bounds_for_naip(raster_dataset)
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
          way_bitmap[p[1]][p[0]] = 1
  print "CACHING way_bitmap numpy array to", DEFAULT_WAY_BITMAP_NPY_FILE
  numpy.save(DEFAULT_WAY_BITMAP_NPY_FILE, way_bitmap)
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

def bounds_for_naip(raster_dataset):
  '''
      clip the NAIP to LEFT_X to RIGHT_X, BOTTOM_Y to TOP_Y
  '''
  sw = pixelToLatLng(raster_dataset, LEFT_X, BOTTOM_Y)
  ne = pixelToLatLng(raster_dataset, RIGHT_X, TOP_Y)
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

def save_naip_as_jpeg(raster_data_path, way_bitmap, training_labels, test_labels, path=None):
  '''
      save the source TIFF as a JPEG, with labels and data overlaid
  '''
  outfile = path
  if not outfile:
    outfile = os.path.splitext(raster_data_path)[0] + ".png"
  im = Image.open(raster_data_path)
  print "GENERATING PNG for %s" % raster_data_path
  rows = len(way_bitmap)
  cols = len(way_bitmap[0])

  # TIFF to JPEG bit from: from: http://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python
  r, g, b, ir = im.split()
  for x in range(cols):
    for y in range(rows):
      r.putpixel((x, y),(255))
  im = Image.merge("RGBA", (ir, ir, ir, r))
  
  # shade training labels
  for label in training_labels:
    start_x = label[1][0]
    start_y = label[1][1]
    for x in range(start_x, start_x+TILE_SIZE):
      for y in range(start_y, start_y+TILE_SIZE):
        r, g, b, a = im.getpixel((x, y))
        if has_ways(label[0]):
          im.putpixel((x, y), (r, g, b, 255))
        else:
          im.putpixel((x, y), (r, g, 255, 255))

  # shade test labels
  for label in test_labels:
    start_x = label[1][0]
    start_y = label[1][1]
    for x in range(start_x, start_x+TILE_SIZE):
      for y in range(start_y, start_y+TILE_SIZE):
        r, g, b, a = im.getpixel((x, y))
        if has_ways(label[0]):
          im.putpixel((x, y), (r, g, b, 255))
        else:
          im.putpixel((x, y), (r, 255, b, 255))

  # show raw data that spawned the labels
  for row in range(0, rows):
    for col in range(0, cols):
      if way_bitmap[row][col]:
        im.putpixel((col, row), (255,0,0, 255))

  im.save(outfile, "PNG")

def download_and_tile_pbf(raster_data_path, raster_dataset, rows, cols):
  '''
      download a certain PBF file from geofabrik unless it exists locally already,
      tile it into training/test labels to match the NAIP image tiles
  '''
  waymap = WayMap()
  file_path = os.path.join(GEO_DATA_DIR, 'district-of-columbia-latest.osm.pbf')
  if not os.path.exists(file_path):
    file_path = download_file('http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf')
  waymap.run_extraction(file_path)

  labels_bitmap = empty_tile_matrix(rows, cols)
  test_labels = []
  training_labels = []
  way_bitmap_npy = numpy.asarray(way_bitmap_for_naip(waymap.extracter.ways, raster_dataset, rows, cols))

  for row in range(TOP_Y, BOTTOM_Y-TILE_SIZE, TILE_SIZE):
    for col in range(LEFT_X, RIGHT_X-TILE_SIZE, TILE_SIZE):
      new_tile = way_bitmap_npy[row:row+TILE_SIZE, col:col+TILE_SIZE]
      if has_ways(new_tile):
        for r in range(row,row+TILE_SIZE):
          for c in range(col,col+TILE_SIZE):
            labels_bitmap[r][c] = 1
      if row < (TOP_Y + (BOTTOM_Y-TOP_Y)*PERCENT_FOR_TRAINING_DATA):
        training_labels.append((new_tile,(col, row)))
      else:
        test_labels.append((new_tile,(col, row)))

  return way_bitmap_npy, labels_bitmap, training_labels, test_labels

def format_as_onehot_arrays(training_labels, test_labels):
  '''
     each label gets converted from an NxN tile with,
     into a one hot array of whether the tile contains ways (i.e. [0,1] or [1,0] for each)
  '''
  onehot_test_labels = []
  for label in test_labels:
    if has_ways(label[0]):
      onehot_test_labels.append([0,1])
    else:
      onehot_test_labels.append([1,0])

  onehot_training_labels = []
  for label in training_labels:
    if has_ways(label[0]):
      onehot_training_labels.append([0,1])
    else:
      onehot_training_labels.append([1,0])

  print "ONE HOT for way presence - {} test labels and {} training labels in".format(len(onehot_training_labels), len(onehot_test_labels))
  return onehot_training_labels, onehot_test_labels


def has_ways(tile):
  '''
     returns true if any pixel on the NxN tile is set to 1
  '''
  for col in range(0, len(tile)):
    for row in range(0, len(tile[col])):
      if tile[col][row] == 1:
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


if __name__ == '__main__':
  
  # dowload and tile NAIP into images to label
  naiper = NAIPDownloader()
  raster_data_path = naiper.download_naips()
  training_images, test_images, raster_dataset, bands_data = read_naip(raster_data_path)
  rows = bands_data.shape[0]
  cols = bands_data.shape[1]
  
  # download and tile labels from PBF file
  way_bitmap, \
  labels_bitmap, \
  training_labels, \
  test_labels = download_and_tile_pbf(raster_data_path, raster_dataset, rows, cols)

  test_images, test_labels = shuffle_in_unison(test_images, test_labels)
  training_images, training_labels = shuffle_in_unison(training_images, training_labels)

  # this step can take a long time, especially for the whole image or a large chunk
  '''
  save_naip_as_jpeg(raster_data_path, 
                    way_bitmap, 
                    training_labels, 
                    test_labels, path="data/naip/labels.png")
  '''

  tiles = len(training_labels)
  # how to log this better?
  h = len(training_labels[0])
  w = len(training_labels[0])
  print("TRAINING/TEST DATA: shaped the tiff data to {} tiles sized {} x {} from the IR band".format(tiles*2, h, w))

  onehot_training_labels, \
  onehot_test_labels = format_as_onehot_arrays(training_labels, test_labels)

  npy_training_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in training_images])
  npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
  npy_training_labels = numpy.asarray(onehot_training_labels)
  npy_test_labels = numpy.asarray(onehot_test_labels)

  # train and test the neural net
  train_neural_net(TILE_SIZE,
                   npy_training_images, 
                   npy_training_labels, 
                   npy_test_images, 
                   npy_test_labels)


  