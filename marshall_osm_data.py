

'''
    1) download geojson road tiles from mapzen
    2) convert the road geographic linestrings to pixels
    2a) rasterize the roads to pixel matrices for the tiles
    2b) try using 2 pixels for the road width, then fix this with trial/error
    3b) further fix with a training layer to guess come up with the width predicitvely
    3) download corresponding MapQuest imagery for the tiles
    4) train a deep learning net with the roads as labeled data for the imagery
    5) 

    geo help from: https://gist.github.com/tucotuco/1193577


'''

import os, math, urllib, sys, json
from globalmaptiles import GlobalMercator


MAPZEN_VECTOR_TILES_API_KEY = 'vector-tiles-NsMiwBc'

class BoundingBox:
  def __init__(self):
    self.northeast = Coordinate()
    self.southwest = Coordinate()


class Coordinate:
  def __init__(self, lat=-999, lon=-999):
    self.lat = lat
    self.lon = lon

  def __str__(self):
    return "{} {}".format(self.lat, self.lon)


class MercatorTile:
  def __init__(self, x=-1, y=-1, z=-1):
    self.x = x
    self.y = y 
    self.z = z

  def __str__(self):
    return "{} {} {}".format(self.x, self.y, self.z)


class Pixel:
  def __init__(self, x=0, y=0):
    self.x = x
    self.y = y

  def __str__(self):
    return "{} {}".format(self.x, self.y)


class OSMDataNormalizer:  

  def __init__(self):
    data_dir = "data/"
    self.tile_size = 256
    self.make_directory(data_dir)
    self.vector_tiles_dir = "data/vector-tiles/"
    self.make_directory(self.vector_tiles_dir)

  def make_directory(self, new_dir):
    '''
       make a directory or complain it already exists
    '''
    try:
      os.mkdir(new_dir);
    except:
      print("{} already exists".format(new_dir))

  def tile_with_coordinates_and_zoom(self, coordinates, zoom):
    scale = (1<<zoom);
    normalized_point = self.normalize_pixel_coords(coordinates)
    return MercatorTile(int(normalized_point.lat * scale), 
                        int(normalized_point.lon * scale), 
                        int(zoom)
                       )

  def normalize_pixel_coords(self, coord):
    if coord.lon > 180:
      coord.lon -= 360
    coord.lon /= 360.0
    coord.lon += 0.5
    coord.lat = 0.5 - ((math.log(math.tan((math.pi/4) + ((0.5 * math.pi *coord.lat) / 180.0))) / math.pi) / 2.0)
    return coord    

  def tiles_for_bounding_box(self, bounding_box, zoom):
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

  def download_tile(self, tile):
      url = self.url_for_tile(tile)
      z_dir = self.vector_tiles_dir + str(tile.z)
      y_dir = z_dir + "/" + str(tile.y)
      self.make_directory(z_dir)
      self.make_directory(y_dir)
      format = 'json'
      filename = '{}.{}'.format(tile.x,format)
      download_path = y_dir + "/" + filename
      urllib.urlretrieve (url, download_path)

  def url_for_tile(self, tile, base_url='http://vector.mapzen.com/osm/'):
      layers = 'roads'
      format = 'json'
      api_key = MAPZEN_VECTOR_TILES_API_KEY
      filename = '{}.{}'.format(tile.x,format)
      url = base_url + '{}/{}/{}/{}?api_key={}'.format(layers,tile.z,tile.y,filename,api_key)
      print url
      return url 

  def osm_url_for_tile(self, tile):
      base_url='http://b.tile.thunderforest.com/outdoors/'
      filename = '{}.{}'.format(tile.x,'png')
      url = base_url + '{}/{}/{}'.format(tile.z,tile.y,filename)
      return url 

  def download_geojson(self):
    ''' 
        download geojson tiles for Yosemite Village from mapzen
    '''
    yosemite_village_bb = BoundingBox()
    yosemite_village_bb.northeast.lat = 37.81385
    yosemite_village_bb.northeast.lon = -119.48559
    yosemite_village_bb.southwest.lat = 37.66724
    yosemite_village_bb.southwest.lon = -119.72454
    zoom = 12

    for tile in self.tiles_for_bounding_box(yosemite_village_bb, zoom):
      self.download_tile(tile)

  def process_geojson(self):
    rootdir = self.vector_tiles_dir
    for folder, subs, files in os.walk(rootdir):
      for filename in files:
        with open(os.path.join(folder, filename), 'r') as src:
          features = json.loads(src.read())['features']
          tile_matrix = self.empty_tile_matrix()
          for f in features:
            if f['geometry']['type'] == 'LineString':
              linestring = f['geometry']['coordinates']
              print "adding a line string with {} points".format(len(linestring))
              dir_string = folder.split(self.vector_tiles_dir)
              z, y = dir_string[1].split('/')
              x = filename.split('.')[0]
              tile = MercatorTile(int(x), int(y), int(z))
              line_matrix = self.pixel_matrix_for_linestring(linestring, tile)
              for x in range(0,255):
                for y in range(0,255):
                  tile_matrix[x][y] = tile_matrix[x][y] or line_matrix[x][y] 

          for row in tile_matrix:
            row_str = ''
            for cell in row:
              row_str += str(cell)
            print row_str


  def empty_tile_matrix(self):
    # initialize the array to all zeroes
    tile_matrix = []    
    for x in range(0,self.tile_size):
      tile_matrix.append([])
      for y in range(0,self.tile_size):
        tile_matrix[x].append(0)     
    return tile_matrix


  def pixel_matrix_for_linestring(self, linestring, tile):
    tile_matrix = self.empty_tile_matrix()
    zoom = tile.z
    print self.osm_url_for_tile(tile)
    # set pixel_matrix to 1 for every point between all points on the line string
    gm = GlobalMercator()

    count = 0
    for current_point in linestring:
      if count == len(linestring) - 1:
        break
      next_point = linestring[count+1]
      current_point_obj = Coordinate(current_point[1], current_point[0])
      next_point_obj = Coordinate(next_point[1], next_point[0])
      start_pixel = gm.LatLngToRaster(current_point_obj.lat,
                                      current_point_obj.lon, zoom)
      start_pixel_obj = Pixel(start_pixel[0]%self.tile_size, start_pixel[1]%self.tile_size)
      end_pixel = gm.LatLngToRaster(next_point_obj.lat,
                                    next_point_obj.lon, zoom)
      end_pixel_obj = Pixel(end_pixel[0]%self.tile_size, end_pixel[1]%self.tile_size)
      pixels = self.pixels_between(start_pixel_obj, end_pixel_obj)

      for p in pixels:
        print '{}, {}'.format(p.x,p.y)
        tile_matrix[p.x][p.y] = 1
      count += 1

    return tile_matrix


  def pixels_between(self, start_pixel, end_pixel):
    pixels = []
 
    if end_pixel.x - start_pixel.x == 0:
      for y in range(min(end_pixel.y, start_pixel.y),
                     max(end_pixel.y, start_pixel.y)):
        p = Pixel()
        p.x = end_pixel.x
        p.y = y
        pixels.append(p) 
      return pixels
      
    slope = (end_pixel.y - start_pixel.y)/(end_pixel.x - start_pixel.x)
    offset = end_pixel.y - slope*end_pixel.x
    
    for x in range(min(end_pixel.x, start_pixel.x),
                   max(end_pixel.x, start_pixel.x)):
      p = Pixel()
      p.x = int(x)
      p.y = int(slope*x + offset)
      pixels.append(p) 
    return pixels

odn = OSMDataNormalizer()
#odn.download_geojson()
odn.process_geojson()