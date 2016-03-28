# cribbed from http://www.machinalis.com/blog/python-for-geospatial-data-processing/

import numpy as np
import os

from osgeo import gdal

def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
              projection, target_value=1):
  """Rasterize the given vector (wrapper for gdal.RasterizeLayer)."""
  data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
  layer = data_source.GetLayer(0)
  driver = gdal.GetDriverByName('MEM')  # In memory dataset
  target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
  target_ds.SetGeoTransform(geo_transform)
  target_ds.SetProjection(projection)
  gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
  return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
  """Rasterize the vectors in the given directory in a single image."""
  labeled_pixels = np.zeros((rows, cols))
  for i, path in enumerate(file_paths):
    label = i+1
    ds = create_mask_from_vector(path, cols, rows, geo_transform,
                   projection, target_value=label)
    band = ds.GetRasterBand(1)
    labeled_pixels += band.ReadAsArray()
    ds = None
  return labeled_pixels


def write_geotiff(fname, data, geo_transform, projection):
  """Create a GeoTIFF file with the given data."""
  driver = gdal.GetDriverByName('GTiff')
  rows, cols = data.shape
  dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
  dataset.SetGeoTransform(geo_transform)
  dataset.SetProjection(projection)
  band = dataset.GetRasterBand(1)
  band.WriteArray(data)
  dataset = None  # Close the file

def read_naip(file_path):
  raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
  geo_transform = raster_dataset.GetGeoTransform()
  proj = raster_dataset.GetProjectionRef()
  bands_data = []
  for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

  bands_data = np.dstack(bands_data)
  rows, cols, n_bands = bands_data.shape

  raster_data_path = 'data/naip/m_3807708_ne_18_1_20130924.tif'
  print("this tiff has {} rows, {} cols, and {} bands".format(rows, cols, n_bands))

def tile_naip(file_path):

  # use gdal_translate to clip an area
  # https://trac.osgeo.org/gdal/wiki/FAQRaster#HowdoIusegdal_translatetoextractorclipasub-sectionofaraster
