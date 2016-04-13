import numpy, os
from osgeo import gdal
from PIL import Image
from pyproj import Proj, transform
from extract_ways import WayMap, download_file
from download_naips import NAIPDownloader
from geo_util import latLonToPixel, pixelToLatLng
from label_chunks_cnn import train_neural_net

tile_size = 12
top_y = 2500
bottom_y = 6800
left_x = 500
right_x = 4300

GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR") # set in Dockerfile as env variable
DEFAULT_WAY_BITMAP_NPY_FILE = os.path.join(GEO_DATA_DIR, )

def read_naip(file_path):
  '''
      from http://www.machinalis.com/blog/python-for-geospatial-data-processing/
  '''
  raster_dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
  coord = pixelToLatLng(raster_dataset, 0, 0)
  proj = raster_dataset.GetProjectionRef()
  bands_data = []
  for b in range(1, raster_dataset.RasterCount+1):
    if b == 4:
      band = raster_dataset.GetRasterBand(b)
      bands_data.append(band.ReadAsArray())

  bands_data = numpy.dstack(bands_data)

  training_images, test_images = tile_naip(raster_dataset, bands_data)
  return training_images, test_images, raster_dataset, bands_data.shape[0], bands_data.shape[1]

def tile_naip(raster_dataset, bands_data):
  rows, cols, n_bands = bands_data.shape
  print("OPENED NAIP with {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
  print("GEO-BOUNDS for image is {}".format(bounds_for_naip(raster_dataset, rows, cols)))

  training_tiled_data = []
  test_tiled_data = []
  x = 0
  for col in range(left_x, right_x-tile_size, tile_size):
    for row in range(top_y, bottom_y-tile_size, tile_size):
      new_tile = bands_data[row:row+tile_size, col:col+tile_size,0:1]
      if x%2 == 0:
        training_tiled_data.append(new_tile)
      else:
        test_tiled_data.append(new_tile)
      x += 1

  shaped_training_data = numpy.array(training_tiled_data)
  shaped_test_data = numpy.array(test_tiled_data)
  tiles, h, w, bands = shaped_training_data.shape
  print("TRAINING DATA: shaped the tiff data to {} tiles sized {} x {} x {}".format(tiles, h, w, bands))
  tiles, h, w, bands = shaped_test_data.shape
  print("TEST DATA: shaped the tiff data to {} tiles sized {} x {} x {}".format(tiles, h, w, bands))
  return shaped_training_data, shaped_test_data

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

  way_bitmap = empty_tile_matrix(cols, rows)
  bounds = bounds_for_naip(raster_dataset, rows, cols)
  ways_on_naip = []
  for way in ways:
    for point_tuple in way['linestring']:
      #print(bounds, point_tuple, bounds_contains_node(bounds, point_tuple))
      if bounds_contains_node(bounds, point_tuple):
        ways_on_naip.append(way)
        break
  print("EXTRACTED {} highways that overlap the NAIP, out of {} ways in the PBF".format(len(ways_on_naip), len(ways)))

  for w in ways_on_naip:
    for x in range(len(w['linestring'])-1):
      current_point = w['linestring'][x]
      next_point = w['linestring'][x+1]
      if not bounds_contains_node(bounds, current_point) or \
         not bounds_contains_node(bounds, next_point):
        continue
      current_pix = latLonToPixel(raster_dataset, current_point)
      next_pix = latLonToPixel(raster_dataset, next_point)
      pixel_line = pixels_between(current_pix, next_pix, cols)
      for p in pixel_line:
        if p[0] < 0 or p[1] < 0 or p[0] >= cols or p[1] >= rows:
          continue
        else:
          way_bitmap[p[0]][p[1]] = 1
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

def bounds_for_naip(raster_dataset, rows, cols):
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

def bounds_contains_node(bounds, point_tuple):
  if point_tuple[0] > bounds['ne'][0]:
    return False
  if point_tuple[0] < bounds['sw'][0]:
    return False
  if point_tuple[1] > bounds['ne'][1]:
    return False
  if point_tuple[1] < bounds['sw'][1]:
    return False
  return True

def save_naip_as_jpeg(raster_data_path, way_bitmap):
  '''
      http://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python
  '''
  outfile = os.path.splitext(raster_data_path)[0] + ".jpg"
  try:
      im = Image.open(raster_data_path)
      r, g, b, ir = im.split()
      im = Image.merge("RGB", (ir,ir,ir))
      print "GENERATING JPEG for %s" % raster_data_path
      rows = len(way_bitmap)
      cols = len(way_bitmap[0])
      for row in range(0, rows):
        for col in range(0, cols):
          if way_bitmap[row][col]:
            im.putpixel((col, row), (255,0,0))
          elif row > top_y and row < bottom_y and col > left_x and col < right_x:
            r, g, b = im.getpixel((col, row))
            im.putpixel((col, row), (int(r*.2),int(g*.2),int(b*.2)))
      im.save(outfile, "JPEG")

  except Exception, e:
      print e

def download_and_tile_pbf(raster_data_path, raster_dataset, rows, cols):
  waymap = WayMap()
  file_path = os.path.join(GEO_DATA_DIR, 'district-of-columbia-latest.osm.pbf')
  if not os.path.exists(file_path):
    file_path = download_file('http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf')
  waymap.run_extraction(file_path)
  way_bitmap = numpy.asarray(way_bitmap_for_naip(waymap.extracter.ways, raster_dataset, rows, cols))
  #save_naip_as_jpeg(raster_data_path, way_bitmap)
  return labels_for_bitmap(way_bitmap, tile_size, rows, cols)

def labels_for_bitmap(way_bitmap, tile_size, rows, cols):
  labels_bmp = empty_tile_matrix(rows, cols)
  test_labels = []
  training_labels = []
  x = 0
  for row in range(top_y, bottom_y-tile_size, tile_size):
    for col in range(left_x, right_x-tile_size, tile_size):
      new_tile = way_bitmap[col:col+tile_size, row:row+tile_size]
      if has_ways(new_tile):
        for c in range(col,col+tile_size):
          for r in range(row,row+tile_size):
            labels_bmp[r][c] = 1
      if x%2 == 0:
        training_labels.append(new_tile)
      else:
        test_labels.append(new_tile)
      x += 1

  onehot_test_labels = []
  for label in test_labels:
    if has_ways(label):
      onehot_test_labels.append([0,1])
    else:
      onehot_test_labels.append([1,0])

  onehot_training_labels = []
  for label in training_labels:
    if has_ways(label):
      onehot_training_labels.append([0,1])
    else:
      onehot_training_labels.append([1,0])

  print "ONE HOT for way presence - {} test labels and {} training labels in".format(len(onehot_training_labels), len(onehot_test_labels))
  return labels_bmp, \
         numpy.asarray(onehot_training_labels), \
         numpy.asarray(onehot_test_labels)

def has_ways(tile):
  for col in range(0, len(tile)):
    for row in range(0, len(tile[col])):
      if tile[col][row] == 1:
        return True
  return False

if __name__ == '__main__':
  naiper = NAIPDownloader()
  raster_data_path = naiper.download_naips()
  training_images, test_images, raster_dataset, rows, cols = read_naip(raster_data_path)
  labels_bmp, training_labels, test_labels = download_and_tile_pbf(raster_data_path, raster_dataset, rows, cols)
  save_naip_as_jpeg(raster_data_path, labels_bmp)
  train_neural_net(training_images, training_labels, test_images, test_labels)
