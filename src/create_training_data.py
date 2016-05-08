import argparse
import numpy, os, sys, time, pickle
from random import shuffle
from osgeo import gdal
from PIL import Image
from pyproj import Proj, transform

from download_labels import WayMap, download_and_extract
from download_naips import NAIPDownloader
from geo_util import latLonToPixel, pixelToLatLng

'''
    constants for how to create labels, 
    from OpenStreetMap way (road) info in PBF files
'''
# enough to cover NAIPs around DC/Maryland/Virginia
PBF_FILE_URLS = ['http://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf',
                 'http://download.geofabrik.de/north-america/us/virginia-latest.osm.pbf',
                 'http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf']

# the number of pixels to count as road, 
# on each side of of the centerline pixels
PIXELS_BESIDE_WAYS = 1

# to count an NxN tile as being "On" for roads,
# N*.25 pixels on that tiles must have been classified as roads
PERCENT_OF_TILE_HEIGHT_TO_ACTIVATE = .50

'''
    constants for NAIP imagery to use   
'''
# values to create the S3 bucket path for some maryland NAIPs
# you can get random NAIPS from here, or the exact HARDCODED_NAIP_LIST above
# \todo document how to configure some of these
NAIP_STATE = 'md'
NAIP_YEAR = '2013'
NAIP_RESOLUTION = '1m'
NAIP_SPECTRUM = 'rgbir' 
NAIP_GRID = '38077'

# set this to a value between 1 and 10 or so,
# and unset HARDCODED_NAIP_LIST, to get some different NAIPs
NUMBER_OF_NAIPS = 8

# set this to True for production data science, False for debugging infrastructure
# speeds up downloads and matrix making when False
RANDOMIZE_NAIPS = False

'''
HARDCODED_NAIP_LIST = [
                  'm_3807708_ne_18_1_20130924.tif',
                  'm_3807708_nw_18_1_20130904.tif',
                  'm_3807708_se_18_1_20130924.tif',
                  'm_3807708_se_18_1_20130924.tif',
                  ]
'''
HARDCODED_NAIP_LIST = None


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

def equalize_data(road_labels, naip_tiles, save_clippings):
  wayless_indices = []
  way_indices = []
  for x in range(len(road_labels)):
    tile = road_labels[x][0]
    if has_ways(tile):
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

def save_image_clipping(tile, status):
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tile_size", default='64', help="tile the NAIP and training data into NxN tiles with this dimension")
  parser.add_argument("--bands", default='0001', help="defaults to 0001 for just IR active")
  parser.add_argument("--extract_type", default='highway', help="highway or tennis")
  parser.add_argument("--cache_way_bmp", default=True, help="disable this to create way bitmaps each run")
  parser.add_argument("--clear_way_bmp_cache", default=False, help="enable this to bust the ay_bmp_cache from previous runs")
  parser.add_argument("--save_clippings", default=False, help="save the training data tiles to /data/naip")
  args = parser.parse_args()

  bands_string = args.bands
  band_list = []
  for char in bands_string:
    band_list.append(int(char))

  raster_data_paths = NAIPDownloader(NUMBER_OF_NAIPS,
                                     RANDOMIZE_NAIPS,
                                     NAIP_STATE,
                                     NAIP_YEAR,
                                     NAIP_RESOLUTION,
                                     NAIP_SPECTRUM,
                                     NAIP_GRID,
                                     HARDCODED_NAIP_LIST).download_naips()  

  tile_size = int(args.tile_size)
  road_labels, naip_tiles, waymap, way_bitmap_npy = random_training_data(raster_data_paths, args.cache_way_bmp, args.clear_way_bmp_cache, args.extract_type, band_list, tile_size)
  equal_count_way_list, equal_count_tile_list = equalize_data(road_labels, naip_tiles, args.save_clippings)
  test_labels, training_labels, test_images, training_images = split_train_test(equal_count_tile_list,equal_count_way_list)
  label_types =  waymap.extracter.types

  cache_path = '/data/cache/'
  try:
    os.mkdir(cache_path);
  except:
    pass
  with open(cache_path + 'training_images.pickle', 'w') as outfile:
    pickle.dump(training_images, outfile)
  with open(cache_path + 'training_labels.pickle', 'w') as outfile:
    pickle.dump(training_labels, outfile)
  with open(cache_path + 'test_images.pickle', 'w') as outfile:
    pickle.dump(test_images, outfile)
  with open(cache_path + 'test_labels.pickle', 'w') as outfile:
    pickle.dump(test_labels, outfile)
  with open(cache_path + 'label_types.pickle', 'w') as outfile:
    pickle.dump(label_types, outfile)
  with open(cache_path + 'raster_data_paths.pickle', 'w') as outfile:
    pickle.dump(raster_data_paths, outfile)
  with open(cache_path + 'way_bitmap_npy.pickle', 'w') as outfile:
    pickle.dump(way_bitmap_npy, outfile)

