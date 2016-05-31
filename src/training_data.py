"""Create training data for a neural net, from NAIP images and OpenStreetMap data."""

from __future__ import print_function

import numpy
import os
import pickle
import random
import sys
import time

from osgeo import gdal
from PIL import Image

from openstreetmap_labels import download_and_extract
from geo_util import lat_lon_to_pixel, pixel_to_lat_lon
from naip_images import NAIP_DATA_DIR

# there is a 300 pixel buffer around NAIPs to be trimmed off, where NAIPs overlap...
# otherwise using overlapping images makes wonky train/test splits
NAIP_PIXEL_BUFFER = 300

# where training data gets cached/retrieved
CACHE_PATH = '/DeepOSM/data/cache/'


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
    cache_filename = raster_data_path + '-ways.bitmap.npy'

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
                               pixels_to_fatten_roads, label_data_files, tile_overlap):
    """Return lists of training images and matching labels."""
    # tile images and labels
    waymap = download_and_extract(label_data_files, extract_type)

    for raster_data_path in raster_data_paths:

        path_parts = raster_data_path.split('/')
        filename = path_parts[len(path_parts) - 1]
        labels_path = CACHE_PATH + filename + '-labels.npy'
        images_path = CACHE_PATH + filename + '-images.npy'
        if os.path.exists(labels_path) and os.path.exists(images_path):
            print("TRAINING DATA for {} already tiled".format(filename))
            continue

        road_labels = []
        naip_tiles = []
        raster_dataset, bands_data = read_naip(raster_data_path, band_list)
        rows = bands_data.shape[0]
        cols = bands_data.shape[1]

        way_bitmap_npy = way_bitmap_for_naip(waymap.extracter.ways, raster_data_path,
                                             raster_dataset, rows, cols, pixels_to_fatten_roads)

        left_x, right_x = NAIP_PIXEL_BUFFER, cols - NAIP_PIXEL_BUFFER
        top_y, bottom_y = NAIP_PIXEL_BUFFER, rows - NAIP_PIXEL_BUFFER

        # tile the way bitmap
        for col in range(left_x, right_x, tile_size / tile_overlap):
            for row in range(top_y, bottom_y, tile_size / tile_overlap):
                if row + tile_size < bottom_y and col + tile_size < right_x:
                    new_tile = way_bitmap_npy[row:row + tile_size, col:col + tile_size]
                    road_labels.append((new_tile, col, row, raster_data_path))

        # tile the NAIP
        for tile in tile_naip(raster_data_path, raster_dataset, bands_data, band_list, tile_size,
                              tile_overlap):
            naip_tiles.append(tile)

        assert len(naip_tiles) == len(road_labels)

        # dump the tiled labels from the way bitmap to disk
        with open(labels_path, 'w') as outfile:
            numpy.save(outfile, numpy.asarray(road_labels))

        # dump the tiled images from the NAIP to disk
        with open(images_path, 'w') as outfile:
            numpy.save(outfile, numpy.asarray(naip_tiles))


def shuffle_in_unison(a, b):
    """See www.stackoverflow.com/questions/11765061/better-way-to-shuffle-two-related-lists."""
    a_shuf = []
    b_shuf = []
    index_shuf = range(len(a))
    random.shuffle(index_shuf)
    for i in index_shuf:
        a_shuf.append(a[i])
        b_shuf.append(b[i])
    return a_shuf, b_shuf


def equalize_data(road_labels, naip_tiles, save_clippings):
    """Make sure labeled data includes an equal set of ON and OFF tiles."""
    road_labels, naip_tiles = shuffle_in_unison(road_labels, naip_tiles)
    wayless_indices = []
    way_indices = []
    for x in range(len(road_labels)):
        tile = road_labels[x][0]
        if has_ways_in_center(tile, 1):
            way_indices.append(x)
        elif not has_ways_in_center(tile, 16):
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


def save_image_clipping(tile, status):
    """Save a tile of training data to disk to visualize."""
    rgbir_matrix = tile[0]
    tile_height = len(rgbir_matrix)

    r_img = numpy.empty([tile_height, tile_height])
    for x in range(len(rgbir_matrix)):
        for y in range(len(rgbir_matrix[x])):
            r_img[x][y] = rgbir_matrix[x][y][0]

    g_img = numpy.empty([tile_height, tile_height])
    for x in range(len(rgbir_matrix)):
        for y in range(len(rgbir_matrix[x])):
            if len(rgbir_matrix[x][y]) > 1:
                g_img[x][y] = rgbir_matrix[x][y][1]
            else:
                g_img[x][y] = rgbir_matrix[x][y][0]

    b_img = numpy.empty([tile_height, tile_height])
    for x in range(len(rgbir_matrix)):
        for y in range(len(rgbir_matrix[x])):
            if len(rgbir_matrix[x][y]) > 2:
                b_img[x][y] = rgbir_matrix[x][y][2]
            else:
                b_img[x][y] = rgbir_matrix[x][y][0]

    im = Image.merge('RGB',
                     (Image.fromarray(r_img).convert('L'), Image.fromarray(g_img).convert('L'),
                      Image.fromarray(b_img).convert('L')))
    outfile_path = tile[2] + '-' + status + '-' + str(tile[1][0]) + ',' + str(tile[1][
        1]) + '-' + '.jpg'
    im.save(outfile_path, "JPEG")


def split_train_test(equal_count_tile_list, equal_count_way_list, percent_for_training_data):
    """Allocate percent_for_training_data for train, and the rest for test."""
    test_labels = []
    training_labels = []
    test_images = []
    training_images = []

    for x in range(0, len(equal_count_way_list)):
        if percent_for_training_data > float(x) / len(equal_count_tile_list):
            training_images.append(equal_count_tile_list[x])
            training_labels.append(equal_count_way_list[x])
        else:
            test_images.append(equal_count_tile_list[x])
            test_labels.append(equal_count_way_list[x])
    return test_labels, training_labels, test_images, training_images


def format_as_onehot_arrays(labels):
    """Return a list of one-hot array labels, for a list of tiles.

    Converts to a one-hot array of whether the tile has ways (i.e. [0,1] or [1,0] for each).
    """
    print("CREATING ONE-HOT LABELS...")
    t0 = time.time()
    on_count = 0
    off_count = 0
    onehot_labels = []
    for label in labels:
        if has_ways_in_center(label[0], 1):
            onehot_labels.append([0, 1])
            on_count += 1
        elif not has_ways_in_center(label[0], 16):
            onehot_labels.append([1, 0])
            off_count += 1

    print("ONE-HOT labels: {} on, {} off ({:.1%} on)".format(on_count, off_count, on_count / float(
        len(labels))))
    print("one-hotting took {0:.1f}s".format(time.time() - t0))
    return onehot_labels


def load_training_tiles(naip_path):
    """Return the image and label tiles for the naip_path."""
    print("LOADING DATA: reading from disk and unpickling")
    t0 = time.time()
    path_parts = naip_path.split('/')
    filename = path_parts[len(path_parts) - 1]
    labels_path = CACHE_PATH + filename + '-labels.npy'
    images_path = CACHE_PATH + filename + '-images.npy'
    try:
        with open(labels_path, 'r') as infile:
            training_labels = numpy.load(infile)
        with open(images_path, 'r') as infile:
            training_images = numpy.load(infile)
    except:
        print("WARNING, skipping file because pickled data bad for {}".format(naip_path))
        return [], []
    print("DATA LOADED: time to deserialize test data {0:.1f}s".format(time.time() - t0))
    return training_labels, training_images


def cache_paths(raster_data_paths):
    """Cache a list of naip image paths, to pass on to the train_neural_net script."""
    try:
        os.mkdir(CACHE_PATH)
    except:
        pass
    with open(CACHE_PATH + 'raster_data_paths.pickle', 'w') as outfile:
        pickle.dump(raster_data_paths, outfile)


def tag_with_locations(test_images, predictions, tile_size):
    """Combine image data with label data, so info can be rendered in a map and list UI.

    Add location data for convenience too.
    """
    combined_data = []
    for idx, img_loc_tuple in enumerate(test_images):
        raster_dataset = gdal.Open(os.path.join(NAIP_DATA_DIR, img_loc_tuple[2]), gdal.GA_ReadOnly)
        ne_lat, ne_lon = pixel_to_lat_lon(raster_dataset, img_loc_tuple[1][0] * tile_size +
                                          tile_size, img_loc_tuple[1][1] * tile_size)
        sw_lat, sw_lon = pixel_to_lat_lon(raster_dataset, img_loc_tuple[1][0] * tile_size,
                                          img_loc_tuple[1][1] * tile_size + tile_size)
        new_tuple = (img_loc_tuple[0], img_loc_tuple[1], img_loc_tuple[2], predictions[idx],
                     ne_lat, ne_lon, sw_lat, sw_lon)
        print(new_tuple)
        combined_data.append(new_tuple)
    return combined_data


if __name__ == "__main__":
    print("Use bin/create_training_data.py instead of running this script.", file=sys.stderr)
    sys.exit(1)
