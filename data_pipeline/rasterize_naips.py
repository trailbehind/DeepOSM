import numpy as np
import os
from osgeo import gdal
from pyproj import Proj, transform

def read_naip(file_path):
  ''' 
      cribbed from http://www.machinalis.com/blog/python-for-geospatial-data-processing/
  '''
  raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
  coord = pixelTocoord(raster_dataset, 0, 0)
  print("this NAIP top left corner is {}".format(coord))
  proj = raster_dataset.GetProjectionRef()
  bands_data = []
  for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

  bands_data = np.dstack(bands_data)
  shaped_data = tile_naip(raster_dataset, bands_data)

def pixelToLatLng(raster_dataset, col, row):
  '''
      Returns lat lng for pixel in epsg:4326
  '''
  # http://gis.stackexchange.com/questions/53617/how-to-find-lat-lon-values-for-every-pixel-in-a-geotiff-file
  # unravel GDAL affine transform parameters
  c, a, b, f, d, e = raster_dataset.GetGeoTransform()
  xp = a * col + b * row + a * 0.5 + b * 0.5 + c
  yp = d * col + e * row + d * 0.5 + e * 0.5 + f

  # http://gis.stackexchange.com/questions/78838/how-to-convert-projected-coordinates-to-lat-lon-using-python
  inProj = Proj(init='epsg:3857')
  outProj = Proj(init='epsg:4326')
  x1,y1 = xp, yp
  x2,y2 = transform(inProj,outProj,x1,y1)
  return(x2, x2)

def tile_naip(raster_dataset, bands_data):
  rows, cols, n_bands = bands_data.shape
  print("this NAIP has {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
  tiled_data = []
  tile_size = 32
  # this code might be inefficient, maybe i'll care later, YOLO
  for row in xrange(0, rows-tile_size, tile_size):
    for col in xrange(0, cols-tile_size, tile_size):
      new_tile = bands_data[row:row+tile_size, col:col+tile_size,0:n_bands]
      tiled_data.append(new_tile)
      print(pixelToLatLng(raster_dataset, row, col))

  shaped_data = np.array(tiled_data)
  tiles, h, w, bands = shaped_data.shape
  print("this shaped_data has {} tiles sized {} x {} x {}".format(tiles, h, w, bands))
  return shaped_data


raster_data_path = 'data/naip/m_3807708_ne_18_1_20130924.tif'
read_naip(raster_data_path)