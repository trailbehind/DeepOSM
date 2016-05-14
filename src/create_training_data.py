from __future__ import print_function

import os
import pickle
import sys
import time
from random import shuffle

from osgeo import gdal
from PIL import Image
import numpy

from download_labels import download_and_extract
from geo_util import latLonToPixel, pixelToLatLng

# there is a 300 pixel buffer around NAIPs that should be trimmed off,
# where NAIPs overlap... using overlapping images makes wonky train/test splits
NAIP_PIXEL_BUFFER = 300


def read_naip(file_path, bands_to_use):
    '''
        read a NAIP from disk
        bands_to_use is an array like [0,0,0,1], designating whether to use each band (R, G, B, IR)
        from http://www.machinalis.com/blog/python-for-geospatial-data-processing/
    '''
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
    '''
        cut a 4-band raster image into tiles,
        tiles are cubes - up to 4 bands, and N height x N width based on tile_size
    '''
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
    '''
        generate a matrix of size rows x cols, initialized to all zeroes,
        but set to 1 for any pixel where an OSM way runs over
    '''
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
            current_pix = latLonToPixel(raster_dataset, current_point)
            next_pix = latLonToPixel(raster_dataset, next_point)
            add_pixels_between(current_pix, next_pix, cols, rows, way_bitmap,
                               pixels_to_fatten_roads)
    print(" {0:.1f}s".format(time.time() - t0))

    print("CACHING %s..." % cache_filename, end="")
    t0 = time.time()
    numpy.save(cache_filename, way_bitmap)
    print(" {0:.1f}s".format(time.time() - t0))

    return way_bitmap


def bounds_for_naip(raster_dataset, rows, cols):
    '''
        clip the NAIP to 0 to cols, 0 to rows
    '''
    left_x, right_x, top_y, bottom_y = \
        NAIP_PIXEL_BUFFER, cols - NAIP_PIXEL_BUFFER, NAIP_PIXEL_BUFFER, rows - NAIP_PIXEL_BUFFER
    sw = pixelToLatLng(raster_dataset, left_x, bottom_y)
    ne = pixelToLatLng(raster_dataset, right_x, top_y)
    return {'sw': sw, 'ne': ne}


def add_pixels_between(start_pixel, end_pixel, cols, rows, way_bitmap, pixels_to_fatten_roads):
    '''
        add the pixels between the start and end to way_bitmap,
        maybe thickened based on config
    '''
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
    '''
        turn on a pixel in way_bitmap if its in bounds
    '''
    if x < NAIP_PIXEL_BUFFER or y < NAIP_PIXEL_BUFFER or x >= len(way_bitmap[
            0]) - NAIP_PIXEL_BUFFER or y >= len(way_bitmap) - NAIP_PIXEL_BUFFER:
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


def random_training_data(raster_data_paths, extract_type, band_list, tile_size,
                         pixels_to_fatten_roads, label_data_files, tile_overlap):
    road_labels = []
    naip_tiles = []

    # tile images and labels
    waymap = download_and_extract(label_data_files, extract_type)
    way_bitmap_npy = {}

    for raster_data_path in raster_data_paths:
        raster_dataset, bands_data = read_naip(raster_data_path, band_list)
        rows = bands_data.shape[0]
        cols = bands_data.shape[1]

        way_bitmap_npy = numpy.asarray(
            way_bitmap_for_naip(waymap.extracter.ways, raster_data_path, raster_dataset, rows, cols,
                                pixels_to_fatten_roads))

        left_x, right_x, top_y, bottom_y = \
            NAIP_PIXEL_BUFFER, cols - NAIP_PIXEL_BUFFER, NAIP_PIXEL_BUFFER, rows - NAIP_PIXEL_BUFFER
        for col in range(left_x, right_x, tile_size / tile_overlap):
            for row in range(top_y, bottom_y, tile_size / tile_overlap):
                if row + tile_size < bottom_y and col + tile_size < right_x:
                    new_tile = way_bitmap_npy[row:row + tile_size, col:col + tile_size]
                    road_labels.append((new_tile, (col, row), raster_data_path))

        for tile in tile_naip(raster_data_path, raster_dataset, bands_data, band_list, tile_size,
                              tile_overlap):
            naip_tiles.append(tile)

    assert len(road_labels) == len(naip_tiles)

    road_labels, naip_tiles = shuffle_in_unison(road_labels, naip_tiles)
    return road_labels, naip_tiles, waymap


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
    center_x = len(tile) / 2
    center_y = len(tile[0]) / 2
    for x in range(center_x - tolerance, center_x + tolerance):
        for y in range(center_y - tolerance, center_y + tolerance):
            pixel_value = tile[x][y]
            if pixel_value != 0:
                return True
    return False


def save_image_clipping(tile, status):
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


def format_as_onehot_arrays(types, training_labels, test_labels):
    '''
        each label gets converted from an NxN tile with way bits flipped,
        into a one hot array of whether the tile contains ways (i.e. [0,1] or [1,0] for each)
    '''
    print("CREATING ONE-HOT LABELS...")
    t0 = time.time()
    print("CREATING TEST one-hot labels")
    onehot_test_labels = onehot_for_labels(test_labels)
    print("CREATING TRAINING one-hot labels")
    onehot_training_labels = onehot_for_labels(training_labels)
    print("one-hotting took {0:.1f}s".format(time.time() - t0))

    return onehot_training_labels, onehot_test_labels


def onehot_for_labels(labels):
    '''
        returns a list of one-hot array labels, for a list of tiles
    '''
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
    return onehot_labels

# where training data gets cached/retrieved
CACHE_PATH = './data/cache/'


def dump_data_to_disk(raster_data_paths, training_images, training_labels, test_images, test_labels,
                      label_types, onehot_training_labels, onehot_test_labels):
    '''
        pickle/json everything, so the analysis app can use the data
    '''
    print("SAVING DATA: pickling and saving to disk")
    t0 = time.time()
    try:
        os.mkdir(CACHE_PATH)
    except:
        pass
    with open(CACHE_PATH + 'training_images.pickle', 'w') as outfile:
        pickle.dump(training_images, outfile)
    with open(CACHE_PATH + 'training_labels.pickle', 'w') as outfile:
        pickle.dump(training_labels, outfile)
    with open(CACHE_PATH + 'test_images.pickle', 'w') as outfile:
        pickle.dump(test_images, outfile)
    with open(CACHE_PATH + 'test_labels.pickle', 'w') as outfile:
        pickle.dump(test_labels, outfile)
    with open(CACHE_PATH + 'label_types.pickle', 'w') as outfile:
        pickle.dump(label_types, outfile)
    with open(CACHE_PATH + 'raster_data_paths.pickle', 'w') as outfile:
        pickle.dump(raster_data_paths, outfile)
    with open(CACHE_PATH + 'onehot_training_labels.pickle', 'w') as outfile:
        pickle.dump(onehot_training_labels, outfile)
    with open(CACHE_PATH + 'onehot_test_labels.pickle', 'w') as outfile:
        pickle.dump(onehot_test_labels, outfile)
    print("SAVE DONE: time to pickle/json and save test data to disk {0:.1f}s".format(time.time() -
                                                                                      t0))


def load_data_from_disk():
    '''
        read training data into memory
    '''
    print("LOADING DATA: reading from disk and unpickling")
    t0 = time.time()
    with open(CACHE_PATH + 'training_images.pickle', 'r') as infile:
        training_images = pickle.load(infile)
    with open(CACHE_PATH + 'training_labels.pickle', 'r') as infile:
        training_labels = pickle.load(infile)
    with open(CACHE_PATH + 'test_images.pickle', 'r') as infile:
        test_images = pickle.load(infile)
    with open(CACHE_PATH + 'test_labels.pickle', 'r') as infile:
        test_labels = pickle.load(infile)
    with open(CACHE_PATH + 'label_types.pickle', 'r') as infile:
        label_types = pickle.load(infile)
    with open(CACHE_PATH + 'raster_data_paths.pickle', 'r') as infile:
        raster_data_paths = pickle.load(infile)
    with open(CACHE_PATH + 'onehot_training_labels.pickle', 'r') as infile:
        onehot_training_labels = pickle.load(infile)
    with open(CACHE_PATH + 'onehot_test_labels.pickle', 'r') as infile:
        onehot_test_labels = pickle.load(infile)
    print("DATA LOADED: time to unpickle/json test data {0:.1f}s".format(time.time() - t0))
    return (raster_data_paths, training_images, training_labels, test_images, test_labels,
            label_types, onehot_training_labels, onehot_test_labels)


if __name__ == "__main__":
    print("Instead of running this file, use bin/create_training_data.py instead.", file=sys.stderr)
    sys.exit(1)
