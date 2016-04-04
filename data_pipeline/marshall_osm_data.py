'''
    1) download geojson road tiles from mapzen
    2) convert the road geographic linestrings to pixels
    2a) rasterize the roads to pixel matrices for the tiles
    2b) chunk the data into thumb_size pieces (break down the 256px tiles into smaller pieces)
    3) download corresponding MapQuest imagery for the tiles
    3a) chunk the imagery into thumb_size pieces

    geo help from: https://gist.github.com/tucotuco/1193577
'''

import urllib.request
import os, math, sys, json
import numpy
from PIL import Image, ImageOps
from globalmaptiles import GlobalMercator
from geo_util import *

class OSMDataNormalizer:  

  def __init__(self, mapzen_key=None):

    self.mapzen_key = mapzen_key
    self.tile_size = 256
    
    # the square size to chop the imagery up into for analysis
    self.thumb_size = 256

    # select a random half of tiles for training
    self.train_vector_tiles_dir = self.make_directory("data/train/vector-tiles", full_path=True)
    self.train_raster_tiles_dir = self.make_directory("data/train/raster-tiles", full_path=True)
    
    # select a random half of tiles for testing
    self.test_vector_tiles_dir = self.make_directory("data/test/vector-tiles", full_path=True)
    self.test_raster_tiles_dir = self.make_directory("data/test/raster-tiles", full_path=True)

    # put even tiles in train, odd tiles in test
    self.download_count = 0

  def make_directory(self, new_dir, full_path=False):
    '''
       try to make a new directory
    '''
    
    if full_path:
      path = ''
      for token in new_dir.split('/'):
        path += token + '/'
        try:
          os.mkdir(path);
        except:
          pass
      return path

    try:
      os.mkdir(new_dir);
    except:
      pass
    return new_dir

  def chunk_of_tahoe_bb(self):
    '''
        a swath around Lake Tahoe that includes mountains, roads, and lakes
    '''
    yosemite_village_bb = BoundingBox()
    yosemite_village_bb.northeast.lat = 37.81385
    yosemite_village_bb.northeast.lon = -119.48559
    yosemite_village_bb.southwest.lat = 39.82726
    yosemite_village_bb.southwest.lon = -119.59648
    return yosemite_village_bb


  def default_bounds_to_analyze(self):
    '''
        analyze a chunk around Yosemite Village
    '''
    yosemite_village_bb = BoundingBox()
    yosemite_village_bb.northeast.lat = 39.93449
    yosemite_village_bb.northeast.lon = -119.35616
    yosemite_village_bb.southwest.lat = 37.46724
    yosemite_village_bb.southwest.lon = -120.02454
    return yosemite_village_bb

  def default_zoom(self):
    '''
        analyze tiles at TMS zoom level 14, by default
    '''
    return 17

  def default_vector_tile_base_url(self):
    ''' 
        the default server to get vector data to train on
    '''
    return 'http://vector.mapzen.com/osm/'

  def default_raster_tile_base_url(self):
    ''' 
        the default server to get satellite imagery to analyze
    '''
    return 'http://otile2.mqcdn.com/tiles/1.0.0/sat/'

  def download_tiles(self):
    ''' 
        download raster satellite and geojson tiles for the region to be analyzed
    '''
    bounding_box = self.chunk_of_tahoe_bb()
    zoom = self.default_zoom()
    tile_download_count = 0
    tiles = self.tiles_for_bounding_box(bounding_box, zoom)
    print("DOWNLOADING {} tiles (x2, vector+raster)".format(len(tiles)))
    
    tile_count = 0
    for tile in tiles:
      vector_tiles_dir = self.train_vector_tiles_dir
      raster_tiles_dir = self.train_raster_tiles_dir
      if tile_count % 2 == 0:
        vector_tiles_dir = self.test_vector_tiles_dir
        raster_tiles_dir = self.test_raster_tiles_dir

      vector_path, raster_name = self.download_tile(self.default_vector_tile_base_url(), 
                                                    'json', 
                                                    vector_tiles_dir, 
                                                    tile,
                                                    suffix = '?api_key={}'.format(self.mapzen_key),
                                                    layers = 'roads')

      raster_path, raster_name = self.download_tile(self.default_raster_tile_base_url(), 
                                                   'jpg', 
                                                   raster_tiles_dir, 
                                                   tile)
      
      if not vector_path and raster_path:
        print("FAILED to download vector, deleting {}{}".format(raster_path, raster_name))
        os.remove(raster_path, vector_path)
      if not raster_path and vector_path:
        print("FAILED to download raster, deleting {}{}".format(vector_path, vector_name))
        os.remove(vector_path, vector_name)
      if vector_path and raster_path:
        tile_count += 1
        self.chop_tile(raster_path, raster_name)
    print("DOWNLOADED {} tiles".format(tile_count))
   

  def tiles_for_bounding_box(self, bounding_box, zoom):
    '''
        returns a list of MeractorTiles that intersect the bounding_box
        at the given zoom
    '''
    tile_array = []
    ne_tile = self.tile_with_coordinates_and_zoom(bounding_box.northeast,
                                                  zoom)
    sw_tile = self.tile_with_coordinates_and_zoom(bounding_box.southwest,
                                                  zoom)
      
    min_x = min(ne_tile.x, sw_tile.x)
    min_y = min(ne_tile.y, sw_tile.y)
    max_y = max(ne_tile.y, sw_tile.y)
    max_x = max(ne_tile.x, sw_tile.x)
    for y in range(min_y, max_y):
      for x in range(min_x, max_x):
        new_tile = MercatorTile()
        new_tile.x = x
        new_tile.y = y
        new_tile.z = zoom
        tile_array.append(new_tile)
    return tile_array

  def tile_with_coordinates_and_zoom(self, coordinates, zoom):
    '''
        returns a MeractorTile for the given coordinates and zoom
    '''
    scale = (1<<zoom);
    normalized_point = self.normalize_pixel_coords(coordinates)
    return MercatorTile(int(normalized_point.lat * scale), 
                        int(normalized_point.lon * scale), 
                        int(zoom)
                       )

  def normalize_pixel_coords(self, coord):
    '''
        convert lat lon to TMS meters
    '''
    if coord.lon > 180:
      coord.lon -= 360
    coord.lon /= 360.0
    coord.lon += 0.5
    coord.lat = 0.5 - ((math.log(math.tan((math.pi/4) + ((0.5 * math.pi *coord.lat) / 180.0))) / math.pi) / 2.0)
    return coord    

  def download_tile(self, base_url, format, directory, tile, suffix='', layers=None):
    '''
        download a map tile from a TMS server
    '''
    url = self.url_for_tile(base_url, format, tile, suffix, layers)
    z_dir = directory + str(tile.z)
    y_dir = z_dir + "/" + str(tile.y)
    self.make_directory(z_dir)
    self.make_directory(y_dir)
    filename = '{}.{}'.format(tile.x,format)
    download_path = y_dir + "/"

    # skip files we already have
    if os.path.exists(download_path + filename):   
      print('SKIPPING, already downloaded: ' + url)
      return download_path, filename

    print('DOWNLOADING: ' + url)
    urllib.request.urlretrieve (url, download_path + filename)
    if not os.path.exists(download_path + filename):   
      return None, None

    return download_path, filename

  def chop_tile(self, path, filename):
    if self.thumb_size == self.tile_size:
      return
    subdir = path + filename.split('.')[0]
    try:
      os.mkdir(subdir);
    except:
      pass

    height = self.thumb_size 
    width = self.thumb_size
    input = path + filename
    im = Image.open(input)
    imgwidth, imgheight = im.size

    img_count = 0
    for y in range(int(self.tile_size/self.thumb_size)):
      for x in range(int(self.tile_size/self.thumb_size)):
        box = (x*self.thumb_size, y*self.thumb_size, x*self.thumb_size+self.thumb_size, y*self.thumb_size+self.thumb_size)
        a = im.crop(box)
        chunk_path = subdir + '/' + str(img_count) + '.jpg'
        if (img_count < 10):
          chunk_path = subdir + '/' + '0000' + str(img_count) + '.jpg'
        elif (img_count < 100):
          chunk_path = subdir + '/' + '000' + str(img_count) + '.jpg'
        elif (img_count < 1000):
          chunk_path = subdir + '/' + '00' + str(img_count) + '.jpg'
        elif (img_count < 10000):
          chunk_path = subdir + '/' + '0' + str(img_count) + '.jpg'
        a.save(chunk_path)
        img_count += 1
    os.remove(path + filename)

  def url_for_tile(self, base_url, format, tile, suffix='', layers=None):
    '''
        compose a URL for a TMS server
    '''
    filename = '{}.{}'.format(tile.x,format)
    url = base_url 
    if layers:
      url += '{}/'.format(layers)
    url = url + '{}/{}/{}{}'.format(tile.z,tile.y,filename,suffix)
    return url 

  def process_geojson(self):
    '''
        convert geojson vector tiles to 256 x 256 matrices
        matrix is 1 if the pixel has road on it, 0 if not
    '''
    self.process_vectors_in_dir(self.train_vector_tiles_dir)
    self.process_vectors_in_dir(self.test_vector_tiles_dir)

  def process_vectors_in_dir(self, rootdir):

    self.gm = GlobalMercator()

    num_images = self.count_rasters_in_dir(rootdir) * pow(self.tile_size/self.thumb_size,2)
    print ("num_images is {} in {}".format(num_images, rootdir))
    labels = None
    if self.train_vector_tiles_dir == rootdir:
      self.train_labels = numpy.zeros(num_images * 2, dtype=numpy.float32)
      self.train_labels = self.train_labels.reshape(num_images, 2)
      labels = self.train_labels 
    else:
      self.test_labels = numpy.zeros(num_images * 2, dtype=numpy.float32)
      self.test_labels = self.test_labels.reshape(num_images, 2)
      labels = self.test_labels 

    index = 0
    for folder, subs, files in os.walk(rootdir):
      for filename in files:
        if not filename.endswith('.json'):
            continue
        has_ways = False
        with open(os.path.join(folder, filename), 'r') as src:
          linestrings = self.linestrings_for_vector_tile(src)
        tile_matrix = self.empty_tile_matrix()
        tile = self.tile_for_folder_and_filename(folder, filename, rootdir)
        for linestring in linestrings:
          # check if tile has any linestrings to set it's one-hot
          tile_matrix = self.add_linestring_to_matrix(linestring, tile, tile_matrix)
        # self.print_matrix(tile_matrix)
        # print '\n\n\n'
        
        # Now set the one_hot value for this label
        for y in range(int(self.tile_size/self.thumb_size)):
          for x in range(int(self.tile_size/self.thumb_size)):
            for tmy in range (self.thumb_size):
              for tmx in range (self.thumb_size):
                if tile_matrix[tmx][tmy] == 1:
                  has_ways = True

            if has_ways:
              labels[index][0] = 1
            else:
              labels[index][1] = 1

            index += 1

  def process_rasters(self):
    '''
        convert raster satellite tiles to 256 x 256 matrices
        floats represent some color info about each pixel

        help in tensorflow data pipeline from https://github.com/silberman/polygoggles/blob/master/datasets.py
    '''
    self.train_images = self.process_rasters_in_dir(self.train_raster_tiles_dir)
    self.test_images = self.process_rasters_in_dir(self.test_raster_tiles_dir)
    print("analyzing {} training images and {} test images".format(len(self.train_images), len(self.test_images)))

  def process_rasters_in_dir(self, rootdir):
    '''
        descends through a TMS tile structure and converts the images
        to a matrix of dimensions: num_images * width * height, dtype=numpy.uint8
    '''

    height = self.thumb_size
    width = self.thumb_size
    num_images = self.count_rasters_in_dir(rootdir)
    images = numpy.zeros(num_images * width * height, dtype=numpy.uint8)
    images = images.reshape(num_images, height, width)
    
    index = 0
    for folder, subs, files in os.walk(rootdir):
      for filename in files:
        if not filename.endswith('.jpg'):
            continue
        tile = self.tile_for_folder_and_filename(folder, filename, rootdir)
        image_filename = os.path.join(folder, filename)
        with open(image_filename, 'rb') as img_file:
          with Image.open(img_file) as open_pil_img:
            pil_image = open_pil_img.convert("L")
            pil_image = ImageOps.invert(pil_image)
        image_matrix = numpy.asarray(pil_image, dtype=numpy.float32)
        images[index] = image_matrix
        index += 1
    print("Packing {} images to a matrix of size num_images * width * height, dtype=numpy.uint8".format(index))

    # Reshape to add a depth dimension
    return images.reshape(num_images, width, height, 1)

  def count_rasters_in_dir(self, rootdir):
    num_images = 0
    for folder, subs, files in os.walk(rootdir):
      for filename in files:
        num_images += 1
    return num_images


  def tile_for_folder_and_filename(self, folder, filename, directory):
    '''
        the MeractorTile given a path to a file on disk
    '''
    dir_string = folder.split(directory)
    try:
      z, x = dir_string[1].split('/')
      y = filename.split('.')[0]
    except:
      # it's a tile cropping
      z, x, y = dir_string[1].split('/')
    return MercatorTile(int(x), int(y), int(z))

  def linestrings_for_vector_tile(self, file_data):
    '''
        flatten linestrings and multilinestrings in a geojson tile
        to a list of linestrings
    '''
    features = json.loads(file_data.read())['features']
    linestrings = []          
    count = 0
    for f in features:
      if f['geometry']['type'] == 'LineString':
        linestring = f['geometry']['coordinates']
        linestrings.append(linestring)   
      if f['geometry']['type'] == 'MultiLineString':
        for ls in f['geometry']['coordinates']:
          linestrings.append(ls)   
    return linestrings

  def add_linestring_to_matrix(self, linestring, tile, matrix):
    '''
        add a pixel linestring to the matrix for a given tile
    '''
    line_matrix = self.pixel_matrix_for_linestring(linestring, tile)
    for x in range(0,self.tile_size):
      for y in range(0,self.tile_size):
        if line_matrix[x][y]:
          matrix[x][y] = line_matrix[x][y] 
    return matrix

  def print_matrix(self, matrix):
    '''
        print an ascii matrix in cosole
    '''
    for row in numpy.rot90(numpy.fliplr(matrix)):
      row_str = ''
      for cell in row:
        row_str += str(cell)
      print(row_str)

  def empty_tile_matrix(self):
    ''' 
        initialize the array to all zeroes
    '''
    tile_matrix = []    
    for x in range(0,self.tile_size):
      tile_matrix.append([])
      for y in range(0,self.tile_size):
        tile_matrix[x].append(0)     
    return tile_matrix

  def pixel_matrix_for_linestring(self, linestring, tile):
    '''
       set pixel_matrix to 1 for every point between all points on the line string
    '''

    line_matrix = self.empty_tile_matrix()
    zoom = tile.z

    count = 0
    for current_point in linestring:
      if count == len(linestring) - 1:
        break
      next_point = linestring[count+1]
      current_point_obj = Coordinate(current_point[1], current_point[0])
      next_point_obj = Coordinate(next_point[1], next_point[0])
      
      start_pixel = self.fromLatLngToPoint(current_point_obj.lat,
                                      current_point_obj.lon, tile)      
      end_pixel = self.fromLatLngToPoint(next_point_obj.lat,
                                    next_point_obj.lon, tile)
      pixels = self.pixels_between(start_pixel, end_pixel)
      for p in pixels:
        line_matrix[p.x][p.y] = 1
      count += 1

    return line_matrix

  def fromLatLngToPoint(self, lat, lng, current_tile):
    '''
       convert a lat/lng/zoom to a pixel on a self.tile_size sized tile
    '''
    zoom = current_tile.z
    tile_for_point = self.gm.GoogleTileFromLatLng(lat, lng, zoom)
    
    tile_x_offset =  (tile_for_point[0] - current_tile.x) * self.tile_size
    tile_y_offset = (tile_for_point[1] - current_tile.y) * self.tile_size
    
    # http://stackoverflow.com/a/17419232/108512
    _pixelOrigin = Pixel()
    _pixelOrigin.x = self.tile_size / 2.0
    _pixelOrigin.y = self.tile_size / 2.0
    _pixelsPerLonDegree = self.tile_size / 360.0
    _pixelsPerLonRadian = self.tile_size / (2 * math.pi)

    point = Pixel()
    point.x = _pixelOrigin.x + lng * _pixelsPerLonDegree

    # Truncating to 0.9999 effectively limits latitude to 89.189. This is
    # about a third of a tile past the edge of the world tile.
    siny = self.bound(math.sin(self.degreesToRadians(lat)), -0.9999,0.9999)
    point.y = _pixelOrigin.y + 0.5 * math.log((1 + siny) / (1 - siny)) * -_pixelsPerLonRadian

    num_tiles = 1 << zoom
    point.x = int(point.x * num_tiles) + tile_x_offset - current_tile.x * self.tile_size
    point.y = int(point.y * num_tiles) + tile_y_offset - current_tile.y * self.tile_size
    return point

  def degreesToRadians(self, deg):
    '''
        return radians given degrees
    ''' 
    return deg * (math.pi / 180)
    
  def bound(self, val, valMin, valMax):
    '''
        used to cap the TMS bounding box to clip the poles
    ''' 
    res = 0
    res = max(val, valMin);
    res = min(val, valMax);
    return res;
    
  def pixels_between(self, start_pixel, end_pixel):
    '''
        return a list of pixels along the ling from 
        start_pixel to end_pixel
    ''' 
    pixels = []
    if end_pixel.x - start_pixel.x == 0:
      for y in range(min(end_pixel.y, start_pixel.y),
                     max(end_pixel.y, start_pixel.y)):
        p = Pixel()
        p.x = end_pixel.x
        p.y = y
        if self.pixel_is_on_tile(p):
          pixels.append(p) 
      return pixels
      
    slope = (end_pixel.y - start_pixel.y)/float(end_pixel.x - start_pixel.x)
    offset = end_pixel.y - slope*end_pixel.x

    num_points = self.tile_size
    i = 0
    while i < num_points:
      p = Pixel()
      floatx = start_pixel.x + (end_pixel.x - start_pixel.x) * i / float(num_points)
      p.x = int(floatx)
      p.y = int(offset + slope * floatx)
      i += 1
    
      if self.pixel_is_on_tile(p):
        pixels.append(p) 

    return pixels

  def pixel_is_on_tile(self, p):
    '''
        return true of p.x and p.y are >= 0 and < self.tile_size
    '''
    if (p.x >= 0 and p.x < self.tile_size and p.y >= 0 and p.y < self.tile_size):
      return True
    return False