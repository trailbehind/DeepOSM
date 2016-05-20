"""Methods for working with geo/raster data."""

from osgeo import osr


def lat_lon_to_pixel(raster_dataset, location):
    """From zacharybears.com/using-python-to-translate-latlon-locations-to-pixels-on-a-geotiff/."""
    ds = raster_dataset
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srs_lat_lon = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs_lat_lon, srs)
    new_location = [None, None]
    # Change the point locations into the GeoTransform space
    (new_location[1], new_location[0], holder) = ct.TransformPoint(location[1], location[0])
    # Translate the x and y coordinates into pixel values
    x = (new_location[1] - gt[0]) / gt[1]
    y = (new_location[0] - gt[3]) / gt[5]
    return (int(x), int(y))


def pixel_to_lat_lon(raster_dataset, col, row):
    """From zacharybears.com/using-python-to-translate-latlon-locations-to-pixels-on-a-geotiff/."""
    ds = raster_dataset
    gt = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srs_lat_lon = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs, srs_lat_lon)
    ulon = col * gt[1] + gt[0]
    ulat = row * gt[5] + gt[3]
    # Transform the point into the GeoTransform space
    (lon, lat, holder) = ct.TransformPoint(ulon, ulat)
    return (lat, lon)
