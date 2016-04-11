import numpy, os
from osgeo import gdal, osr
from pyproj import Proj, transform
from extract_ways import WayMap, download_file
from download_naips import NAIPDownloader
from PIL import Image

def read_naip(file_path):
  ''' 
      from http://www.machinalis.com/blog/python-for-geospatial-data-processing/
  '''
  raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
  coord = pixelToLatLng(raster_dataset, 0, 0)
  proj = raster_dataset.GetProjectionRef()
  bands_data = []
  for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

  bands_data = numpy.dstack(bands_data)
  shaped_data = tile_naip(raster_dataset, bands_data)

def latLonToPixel(raster_dataset, location):
  '''
      from http://zacharybears.com/using-python-to-translate-latlon-locations-to-pixels-on-a-geotiff/
  '''
  ds = raster_dataset
  gt = ds.GetGeoTransform()
  srs = osr.SpatialReference()
  srs.ImportFromWkt(ds.GetProjection())
  srsLatLong = srs.CloneGeogCS()
  ct = osr.CoordinateTransformation(srsLatLong,srs)
  new_location =[None, None]
  # Change the point locations into the GeoTransform space
  (new_location[1],new_location[0],holder) = ct.TransformPoint(location[1],location[0])
  # Translate the x and y coordinates into pixel values
  x = (new_location[1]-gt[0])/gt[1]
  y = (new_location[0]-gt[3])/gt[5]
  return(int(x),int(y))

def pixelToLatLng(raster_dataset, col, row):
  '''
      from http://zacharybears.com/using-python-to-translate-latlon-locations-to-pixels-on-a-geotiff/
  '''
  ds = raster_dataset
  gt = ds.GetGeoTransform()
  srs = osr.SpatialReference()
  srs.ImportFromWkt(ds.GetProjection())
  srsLatLong = srs.CloneGeogCS()
  ct = osr.CoordinateTransformation(srs,srsLatLong)
  ulon = col*gt[1]+gt[0]
  ulat = row*gt[5]+gt[3]
  # Transform the point into the GeoTransform space
  (lon,lat,holder) = ct.TransformPoint(ulon,ulat)
  return (lat, lon)

def tile_naip(raster_dataset, bands_data):
  rows, cols, n_bands = bands_data.shape
  print("OPENED NAIP with {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
  print("GEO-BOUNDS for image is {}".format(bounds_for_naip(raster_dataset, rows, cols)))

  waymap = WayMap()
  file_path = download_file('http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf')
  waymap.run_extraction(file_path)
  way_bitmap_for_naip(waymap.extracter.ways, raster_dataset, rows, cols)

  tiled_data = []
  tile_size = 32
  # this code might be inefficient, maybe i'll care later, YOLO
  for row in xrange(0, rows-tile_size, tile_size):
    for col in xrange(0, cols-tile_size, tile_size):
      new_tile = bands_data[row:row+tile_size, col:col+tile_size,0:n_bands]
      tiled_data.append(new_tile)
     
  shaped_data = numpy.array(tiled_data)
  tiles, h, w, bands = shaped_data.shape
  print("SHAPED the tiff data to {} tiles sized {} x {} x {}".format(tiles, h, w, bands))
  return shaped_data

#top_y = 2500
#bottom_y = 3000
#left_x = 2500
#right_x = 3000

def bounds_for_naip(raster_dataset, rows, cols):
  sw = pixelToLatLng(raster_dataset, 0, rows-1)
  ne = pixelToLatLng(raster_dataset, cols-1, 0)
  return {'sw': sw, 'ne': ne}

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

def way_bitmap_for_naip(ways, raster_dataset, rows, cols):
  ''' 
    generate a matrix of size rows x cols, initialized to all zeroes,
    but set to 1 for any pixel where an OSM way runs over
  '''
  way_bitmap = empty_tile_matrix(cols, rows)
  bounds = bounds_for_naip(raster_dataset, rows, cols)
  ways_on_naip = []
  for way in ways:
    for point_tuple in way['linestring']:
      if bounds_contains_node(bounds, point_tuple):
        ways_on_naip.append(way)
        break
  print("EXTRACTED {} highways that overlap the NAIP, out of {} ways in the PBF, ".format(len(ways_on_naip), len(ways)))

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
  save_naip_as_jpeg(raster_data_path, way_bitmap)


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
      im = Image.merge("RGB", (r,g,b))
      print "GENERATING JPEG for %s" % raster_data_path
      
      for row in range(0, len(way_bitmap)):
        for col in range(0, len(way_bitmap[row])):
          if way_bitmap[row][col]:
            r, g, b = im.getpixel((row, col))
            im.putpixel((row, col), (255,0,0))
          #elif row > top_y and row < bottom_y and col > left_x and col < right_x:
          else:
            r, g, b = im.getpixel((row, col))
            im.putpixel((row, col), (int(r*.2),int(g*.2),int(b*.2)))
      im.save(outfile, "JPEG")

  except Exception, e:
      print e

if __name__ == '__main__':
  naiper = NAIPDownloader()
  naiper.download_naips()
  raster_data_path = 'data/naip/m_3807708_ne_18_1_20130924.tif'
  read_naip(raster_data_path)