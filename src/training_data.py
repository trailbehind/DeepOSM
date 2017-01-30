"""Create training data for a neural net, from NAIP images and OpenStreetMap data."""

from __future__ import print_function
import numpy
import os
import pickle
import random
import sys
import time
from osgeo import gdal
from openstreetmap_labels import download_and_extract
from geo_util import lat_lon_to_pixel, pixel_to_lat_lon, pixel_to_lat_lon_web_mercator
from naip_images import NAIP_DATA_DIR, NAIPDownloader
from src.config import CACHE_PATH, LABEL_CACHE_DIRECTORY, LABELS_DATA_DIR, IMAGE_CACHE_DIRECTORY, \
    METADATA_PATH

# there is a 300 pixel buffer around NAIPs to be trimmed off, where NAIPs overlap...
# otherwise using overlapping images makes wonky train/test splits
NAIP_PIXEL_BUFFER = 300


def read_naip(file_path, bands_to_use):
    """
    Read in a NAIP, based on www.machinalis.com/blog/python-for-geospatial-data-processing.

    Bands_to_use is an array like [0,0,0,1], designating whether to use each band (R, G, B, IR).
    """
    raster_dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

    bands_data = []
    index = 0
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        if bands_to_use[index] == 1:
            bands_data.append(band.ReadAsArray())
        index += 1
    bands_data = numpy.dstack(bands_data)

    return raster_dataset, bands_data


def tile_naip(raster_data_path, raster_dataset, bands_data, bands_to_use, tile_size, tile_overlap):
    """
    Cut a 4-band raster image into tiles.

    Tiles are cubes - up to 4 bands, and N height x N width based on tile_size.
    """
    on_band_count = 0
    for b in bands_to_use:
        if b == 1:
            on_band_count += 1

    rows, cols, n_bands = bands_data.shape
    print("OPENED NAIP with {} rows, {} cols, and {} bands".format(rows, cols, n_bands))
    print("GEO-BOUNDS for image chunk is {}".format(bounds_for_naip(raster_dataset, rows, cols)))

    all_tiled_data = []

    for col in range(NAIP_PIXEL_BUFFER, cols - NAIP_PIXEL_BUFFER, tile_size / tile_overlap):
        for row in range(NAIP_PIXEL_BUFFER, rows - NAIP_PIXEL_BUFFER, tile_size / tile_overlap):
            if row + tile_size < rows - NAIP_PIXEL_BUFFER and \
               col + tile_size < cols - NAIP_PIXEL_BUFFER:
                new_tile = bands_data[row:row + tile_size, col:col + tile_size, 0:on_band_count]
                all_tiled_data.append((new_tile, (col, row), raster_data_path))

    return all_tiled_data


def way_bitmap_for_naip(
        ways, raster_data_path,
        raster_dataset,
        rows, cols, pixels_to_fatten_roads=None):
    """
    Generate a matrix of size rows x cols, initialized to all zeroes.

    Set matrix to 1 for any pixel where an OSM way runs over.
    """
    parts = raster_data_path.split('/')
    naip_grid = parts[len(parts)-2]
    naip_filename = parts[len(parts)-1]
    cache_filename = LABELS_DATA_DIR + '/' + naip_grid + '/' + naip_filename + '-ways.bitmap.npy'

    try:
        arr = numpy.load(cache_filename)
        print("CACHED: read label data from disk")
        return arr
    except:
        pass
        # print "ERROR reading bitmap cache from disk: {}".format(cache_filename)

    way_bitmap = numpy.zeros([rows, cols], dtype=numpy.int)
    bounds = bounds_for_naip(raster_dataset, rows, cols)
    ways_on_naip = []

    for way in ways:
        for point_tuple in way['linestring']:
            if bounds_contains_point(bounds, point_tuple):
                ways_on_naip.append(way)
                break
    print("EXTRACTED {} highways in NAIP bounds, of {} ways".format(len(ways_on_naip), len(ways)))

    print("MAKING BITMAP for way presence...", end="")
    t0 = time.time()
    for w in ways_on_naip:
        for x in range(len(w['linestring']) - 1):
            current_point = w['linestring'][x]
            next_point = w['linestring'][x + 1]
            if not bounds_contains_point(bounds, current_point) or \
               not bounds_contains_point(bounds, next_point):
                continue
            current_pix = lat_lon_to_pixel(raster_dataset, current_point)
            next_pix = lat_lon_to_pixel(raster_dataset, next_point)
            add_pixels_between(current_pix, next_pix, cols, rows, way_bitmap,
                               pixels_to_fatten_roads)
    print(" {0:.1f}s".format(time.time() - t0))

    print("CACHING %s..." % cache_filename, end="")
    t0 = time.time()
    # make sure cache_filename's parent folder exists
    try:
        os.makedirs(os.path.dirname(cache_filename))
    except:
        pass
    # then save file to cache_filename
    numpy.save(cache_filename, way_bitmap)
    print(" {0:.1f}s".format(time.time() - t0))

    return way_bitmap


def bounds_for_naip(raster_dataset, rows, cols):
    """Clip the NAIP to 0 to cols, 0 to rows."""
    left_x, right_x, top_y, bottom_y = \
        NAIP_PIXEL_BUFFER, cols - NAIP_PIXEL_BUFFER, NAIP_PIXEL_BUFFER, rows - NAIP_PIXEL_BUFFER
    sw = pixel_to_lat_lon(raster_dataset, left_x, bottom_y)
    ne = pixel_to_lat_lon(raster_dataset, right_x, top_y)
    return {'sw': sw, 'ne': ne}


def add_pixels_between(start_pixel, end_pixel, cols, rows, way_bitmap, pixels_to_fatten_roads):
    """Add the pixels between the start and end to way_bitmap, maybe thickened based on config."""
    if end_pixel[0] - start_pixel[0] == 0:
        for y in range(min(end_pixel[1], start_pixel[1]), max(end_pixel[1], start_pixel[1])):
            safe_add_pixel(end_pixel[0], y, way_bitmap)
            # if configged, fatten lines
            for x in range(1, pixels_to_fatten_roads + 1):
                safe_add_pixel(end_pixel[0] - x, y, way_bitmap)
                safe_add_pixel(end_pixel[0] + x, y, way_bitmap)
        return

    slope = (end_pixel[1] - start_pixel[1]) / float(end_pixel[0] - start_pixel[0])
    offset = end_pixel[1] - slope * end_pixel[0]

    i = 0
    while i < cols:
        floatx = start_pixel[0] + (end_pixel[0] - start_pixel[0]) * i / float(cols)
        p = (int(floatx), int(offset + slope * floatx))
        safe_add_pixel(p[0], p[1], way_bitmap)
        i += 1
        # if configged, fatten lines
        for x in range(1, pixels_to_fatten_roads + 1):
            safe_add_pixel(p[0], p[1] - x, way_bitmap)
            safe_add_pixel(p[0], p[1] + x, way_bitmap)
            safe_add_pixel(p[0] - x, p[1], way_bitmap)
            safe_add_pixel(p[0] + x, p[1], way_bitmap)


def safe_add_pixel(x, y, way_bitmap):
    """Turn on a pixel in way_bitmap if its in bounds."""
    if x < NAIP_PIXEL_BUFFER or y < NAIP_PIXEL_BUFFER or x >= len(way_bitmap[
            0]) - NAIP_PIXEL_BUFFER or y >= len(way_bitmap) - NAIP_PIXEL_BUFFER:
        return
    way_bitmap[y][x] = 1


def bounds_contains_point(bounds, point_tuple):
    """Return True if the bounds geographically contains the point_tuple."""
    if point_tuple[0] > bounds['ne'][0]:
        return False
    if point_tuple[0] < bounds['sw'][0]:
        return False
    if point_tuple[1] > bounds['ne'][1]:
        return False
    if point_tuple[1] < bounds['sw'][1]:
        return False
    return True


def create_tiled_training_data(raster_data_paths, extract_type, band_list, tile_size,
                               pixels_to_fatten_roads, label_data_files, tile_overlap, naip_state):
    """Save tiles for training data to disk, file names are padded with 16 0s.

    Images are named 0000000000000000.colors.
    Labels are named 0000000000000000.lbl.
    """
    # tile images and labels
    waymap = download_and_extract(label_data_files, extract_type)

    tile_index = 0

    for raster_data_path in raster_data_paths:

        # TODO need new code to check cache
        raster_dataset, bands_data = read_naip(raster_data_path, band_list)
        rows = bands_data.shape[0]
        cols = bands_data.shape[1]

        way_bitmap_npy = way_bitmap_for_naip(waymap.extracter.ways, raster_data_path,
                                             raster_dataset, rows, cols, pixels_to_fatten_roads)

        left_x, right_x = NAIP_PIXEL_BUFFER, cols - NAIP_PIXEL_BUFFER
        top_y, bottom_y = NAIP_PIXEL_BUFFER, rows - NAIP_PIXEL_BUFFER

        # tile the way bitmap
        origin_tile_index = tile_index
        for col in range(left_x, right_x, tile_size / tile_overlap):
            for row in range(top_y, bottom_y, tile_size / tile_overlap):
                if row + tile_size < bottom_y and col + tile_size < right_x:
                    file_suffix = '{0:016d}'.format(tile_index)
                    label_filepath = "{}/{}.lbl".format(LABEL_CACHE_DIRECTORY, file_suffix)
                    new_tile = way_bitmap_npy[row:row + tile_size, col:col + tile_size]
                    with open(label_filepath, 'w') as outfile:
                        numpy.save(outfile, numpy.asarray((new_tile, col, row, raster_data_path)))
                    tile_index += 1

        tile_index = origin_tile_index
        # tile the NAIP
        for tile in tile_naip(raster_data_path, raster_dataset, bands_data, band_list, tile_size,
                              tile_overlap):
            file_suffix = '{0:016d}'.format(tile_index)
            img_filepath = "{}/{}.colors".format(IMAGE_CACHE_DIRECTORY, file_suffix)
            with open(img_filepath, 'w') as outfile:
                numpy.save(outfile, tile)
            tile_index += 1

    # dump the metadata to disk for configuring the analysis script later
    training_info = {'bands': band_list, 'tile_size': tile_size, 'naip_state': naip_state}
    with open(CACHE_PATH + METADATA_PATH, 'w') as outfile:
        pickle.dump(training_info, outfile)


def equalize_data(road_labels, naip_tiles, save_clippings):
    """Make sure labeled data includes an equal set of ON and OFF tiles."""
    wayless_indices = []
    way_indices = []
    for x in range(0, len(road_labels)):
        if road_labels[x][0] == 0:
            way_indices.append(x)
        else:
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
        equal_count_tile_list.append(naip_tiles[wayless_index])
    return equal_count_way_list, equal_count_tile_list


def has_ways_in_center(tile, tolerance):
    """Return true if the tile has road pixels withing tolerance pixels of the tile center."""
    center_x = len(tile) / 2
    center_y = len(tile[0]) / 2
    for x in range(center_x - tolerance, center_x + tolerance):
        for y in range(center_y - tolerance, center_y + tolerance):
            pixel_value = tile[x][y]
            if pixel_value != 0:
                return True
    return False


def format_as_onehot_arrays(new_label_paths):
    """Return a list of one-hot array labels, for a list of tiles.

    Converts to a one-hot array of whether the tile has ways (i.e. [0,1] or [1,0] for each).
    """
    training_images, onehot_training_labels = [], []
    print("CREATING ONE-HOT LABELS...")
    t0 = time.time()
    on_count = 0
    off_count = 0
    for filename in new_label_paths:

        full_path = "{}/{}".format(LABEL_CACHE_DIRECTORY, filename)
        label = numpy.load(full_path)

        parts = full_path.split('.')[0].split('/')
        file_suffix = parts[len(parts)-1]
        img_path = "{}/{}.colors".format(IMAGE_CACHE_DIRECTORY, file_suffix)

        if has_ways_in_center(label[0], 1):
            onehot_training_labels.append([0, 1])
            on_count += 1
            training_images.append(numpy.load(img_path))
        elif not has_ways_in_center(label[0], 16):
            onehot_training_labels.append([1, 0])
            off_count += 1
            training_images.append(numpy.load(img_path))

    print("one-hotting took {0:.1f}s".format(time.time() - t0))
    return training_images, onehot_training_labels


def load_training_tiles(number_of_tiles):
    """Return number_of_tiles worth of training_label_paths."""
    print("LOADING DATA: reading from disk and unpickling")
    t0 = time.time()
    training_label_paths = []
    all_paths = os.listdir(LABEL_CACHE_DIRECTORY)
    for x in range(0, number_of_tiles):
        label_path = random.choice(all_paths)
        training_label_paths.append(label_path)
    print("DATA LOADED: time to deserialize test data {0:.1f}s".format(time.time() - t0))
    return training_label_paths


def load_all_training_tiles(naip_path, bands):
    """Return the image and label tiles for the naip_path."""
    print("LOADING DATA: reading from disk and unpickling")
    t0 = time.time()
    tile_size = 64
    tile_overlap = 1
    raster_dataset, bands_data = read_naip(naip_path, bands)
    training_images = tile_naip(naip_path, raster_dataset, bands_data, bands, tile_size,
                                tile_overlap)
    rows = bands_data.shape[0]
    cols = bands_data.shape[1]
    parts = naip_path.split('/')
    naip_grid = parts[len(parts)-2]
    naip_filename = parts[len(parts)-1]
    cache_filename = LABELS_DATA_DIR + '/' + naip_grid + '/' + naip_filename + '-ways.bitmap.npy'
    way_bitmap_npy = numpy.load(cache_filename)

    left_x, right_x = NAIP_PIXEL_BUFFER, cols - NAIP_PIXEL_BUFFER
    top_y, bottom_y = NAIP_PIXEL_BUFFER, rows - NAIP_PIXEL_BUFFER

    training_labels = []
    for col in range(left_x, right_x, tile_size / tile_overlap):
        for row in range(top_y, bottom_y, tile_size / tile_overlap):
            if row + tile_size < bottom_y and col + tile_size < right_x:
                new_tile = way_bitmap_npy[row:row + tile_size, col:col + tile_size]
                training_labels.append(numpy.asarray((new_tile, col, row, naip_path)))

    print("DATA LOADED: time to deserialize test data {0:.1f}s".format(time.time() - t0))
    return training_labels, training_images


def tag_with_locations(test_images, predictions, tile_size, state_abbrev):
    """Combine image data with label data, so info can be rendered in a map and list UI.

    Add location data for convenience too.
    """
    combined_data = []
    for idx, img_loc_tuple in enumerate(test_images):
        raster_filename = img_loc_tuple[2]
        raster_dataset = gdal.Open(os.path.join(NAIP_DATA_DIR, raster_filename), gdal.GA_ReadOnly)
        raster_tile_x = img_loc_tuple[1][0]
        raster_tile_y = img_loc_tuple[1][1]
        ne_lat, ne_lon = pixel_to_lat_lon_web_mercator(raster_dataset, raster_tile_x +
                                                       tile_size, raster_tile_y)
        sw_lat, sw_lon = pixel_to_lat_lon_web_mercator(raster_dataset, raster_tile_x,
                                                       raster_tile_y + tile_size)
        certainty = predictions[idx][0]
        formatted_info = {'certainty': certainty, 'ne_lat': ne_lat, 'ne_lon': ne_lon,
                          'sw_lat': sw_lat, 'sw_lon': sw_lon, 'raster_tile_x': raster_tile_x,
                          'raster_tile_y': raster_tile_y, 'raster_filename': raster_filename,
                          'state_abbrev': state_abbrev, 'country_abbrev': 'USA'
                          }
        combined_data.append(formatted_info)
    return combined_data


def download_and_serialize(number_of_naips,
                           randomize_naips,
                           naip_state,
                           naip_year,
                           extract_type,
                           bands,
                           tile_size,
                           pixels_to_fatten_roads,
                           label_data_files,
                           tile_overlap):
    """Download NAIP images, PBF files, and serialize training data."""
    raster_data_paths = NAIPDownloader(number_of_naips,
                                       randomize_naips,
                                       naip_state,
                                       naip_year).download_naips()

    create_tiled_training_data(raster_data_paths,
                               extract_type,
                               bands,
                               tile_size,
                               pixels_to_fatten_roads,
                               label_data_files,
                               tile_overlap,
                               naip_state)
    return raster_data_paths


if __name__ == "__main__":
    print("Use bin/create_training_data.py instead of running this script.", file=sys.stderr)
    sys.exit(1)
