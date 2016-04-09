import numpy as np
import os
from osgeo import gdal, osr
from pyproj import Proj, transform
from extract_ways import WayMap
from download_naips import NAIPDownloader

def read_naip(file_path):
  ''' 
      cribbed from http://www.machinalis.com/blog/python-for-geospatial-data-processing/
  '''
  raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
  coord = pixelToLatLng(raster_dataset, 0, 0)
  proj = raster_dataset.GetProjectionRef()
  bands_data = []
  for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

  bands_data = np.dstack(bands_data)
  shaped_data = tile_naip(raster_dataset, bands_data)


def pixelToLatLng(raster_dataset,col, row):
  '''
      http://zacharybears.com/using-python-to-translate-latlon-locations-to-pixels-on-a-geotiff/
  '''
  # Load the image dataset
  ds = raster_dataset
  # Get a geo-transform of the dataset
  gt = ds.GetGeoTransform()
  # Create a spatial reference object for the dataset
  srs = osr.SpatialReference()
  srs.ImportFromWkt(ds.GetProjection())
  # Set up the coordinate transformation object
  srsLatLong = srs.CloneGeogCS()
  ct = osr.CoordinateTransformation(srs,srsLatLong)
  # Go through all the point pairs and translate them to pixel pairings

  ulon = col*gt[1]+gt[0]
  ulat = row*gt[5]+gt[3]
  # Transform the points to the space
  (lon,lat,holder) = ct.TransformPoint(ulon,ulat)
  # Add the point to our return array
  return (lat, lon)

def tile_naip(raster_dataset, bands_data):
  rows, cols, n_bands = bands_data.shape
  print("this NAIP has {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
  print("this NAIP has bounds {}".format(bounds_for_naip(raster_dataset, rows, cols)))

  waymap = WayMap()
  waymap.run_extraction('./district-of-columbia-latest.osm.pbf')
  way_bitmap_for_naip(waymap.extracter.ways, raster_dataset, rows, cols)
  print(len(waymap.extracter.ways))

  tiled_data = []
  tile_size = 32
  # this code might be inefficient, maybe i'll care later, YOLO
  for row in xrange(0, rows-tile_size, tile_size):
    for col in xrange(0, cols-tile_size, tile_size):
      new_tile = bands_data[row:row+tile_size, col:col+tile_size,0:n_bands]
      tiled_data.append(new_tile)
     
  shaped_data = np.array(tiled_data)
  tiles, h, w, bands = shaped_data.shape
  print("this shaped_data has {} tiles sized {} x {} x {}".format(tiles, h, w, bands))
  return shaped_data

def bounds_for_naip(raster_dataset, rows, cols):
  sw = pixelToLatLng(raster_dataset, 0, cols-1)
  ne = pixelToLatLng(raster_dataset, rows-1, 0)
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
  way_bitmap = empty_tile_matrix(rows, cols)
  bounds = bounds_for_naip(raster_dataset, rows, cols)
  ways_on_naip = []
  for way in ways:
    for point_tuple in way['linestring']:
      if bounds_contains_node(bounds, point_tuple):
        ways_on_naip.append(way)
        break
  print("found {}/{} ways that overlap this naip".format(len(ways_on_naip), len(ways)))

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

naiper = NAIPDownloader()
naiper.download_naips()
raster_data_path = 'data/naip/m_3807708_ne_18_1_20130924.tif'
read_naip(raster_data_path)