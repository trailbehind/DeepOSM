"""A simple 1 layer network."""
from __future__ import division, print_function, absolute_import

import numpy
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d

from src.training_data import load_training_tiles, equalize_data, \
    split_train_test, format_as_onehot_arrays, shuffle_in_unison, has_ways_in_center


def train_on_cached_data(raster_data_paths, neural_net_type, bands, tile_size, number_of_epochs):
    """Load tiled/cached data, which was prepared for the NAIPs listed in raster_data_paths.

    Read in each NAIP's images/labels, add to train/test data, run some epochs as each is added.
    Keep the train and test sets to a max of 10K images by throwing out random data sometimes.
    """
    training_images = []
    onehot_training_labels = []
    test_images = []
    onehot_test_labels = []
    model = None

    for path in raster_data_paths:
        # read in another NAIP worth of data
        labels, images = load_training_tiles(path)
        if len(labels) == 0 or len(images) == 0:
            continue
        equal_count_way_list, equal_count_tile_list = equalize_data(labels, images, False)
        new_test_labels, training_labels, new_test_images, new_training_images = \
            split_train_test(equal_count_tile_list, equal_count_way_list, .9)

        if len(training_labels) == 0:
            print("WARNING: a naip image didn't have any road labels?")
            continue
        if len(new_test_labels) == 0:
            print("WARNING: a naip image didn't have any road images?")
            continue

        # add it to the training and test lists
        [training_images.append(i) for i in new_training_images]
        [test_images.append(i) for i in new_test_images]
        [onehot_training_labels.append(l) for l in format_as_onehot_arrays(training_labels)]
        [onehot_test_labels.append(l) for l in format_as_onehot_arrays(new_test_labels)]

        # once we have 100 test_images, maybe from more than one NAIP, train on a mini batch
        if len(training_images) >= 100:
            # continue training the model with the new data set
            model = train_with_data(onehot_training_labels, onehot_test_labels, test_images,
                                    training_images, neural_net_type, bands, tile_size,
                                    number_of_epochs, model)
            training_images = []
            onehot_training_labels = []

        # keep test list to 10000 images, in case the machine doesn't have much memory
        if len(test_images) > 10000:
            # shuffle so when we chop off data, it's from many NAIPs, not just the last one
            shuffle_in_unison(test_images, onehot_test_labels)
            test_images = test_images[:9000]
            onehot_test_labels = onehot_test_labels[:9000]

    return test_images, model


def train_with_data(onehot_training_labels, onehot_test_labels, test_images, training_images,
                    neural_net_type, band_list, tile_size, number_of_epochs, model):
    """Package data for tensorflow and analyze."""
    npy_training_labels = numpy.asarray(onehot_training_labels)
    npy_test_labels = numpy.asarray(onehot_test_labels)

    # normalize 0-255 values to 0-1
    norm_training_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in training_images])
    norm_train_images = norm_training_images.astype(numpy.float32)
    norm_train_images = numpy.multiply(norm_train_images, 1.0 / 255.0)

    norm_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
    norm_test_images = norm_test_images.astype(numpy.float32)
    norm_test_images = numpy.multiply(norm_test_images, 1.0 / 255.0)

    if not model:
        on_band_count = 0
        for b in band_list:
            if b == 1:
                on_band_count += 1

        network = tflearn.input_data(shape=[None, tile_size, tile_size, on_band_count])
        if neural_net_type == 'one_layer_relu':
            network = tflearn.fully_connected(network, 32, activation='relu')
        elif neural_net_type == 'one_layer_relu_conv':
            network = conv_2d(network, 256, 16, activation='relu')
            network = max_pool_2d(network, 3)
        else:
            print("ERROR: exiting, unknown layer type for neural net")

        # classify as road or not road
        softmax = tflearn.fully_connected(network, 2, activation='softmax')

        # based on parameters from www.cs.toronto.edu/~vmnih/docs/Mnih_Volodymyr_PhD_Thesis.pdf
        momentum = tflearn.optimizers.Momentum(
            learning_rate=.005, momentum=0.9,
            lr_decay=0.0002, name='Momentum')

        net = tflearn.regression(softmax, optimizer=momentum, loss='categorical_crossentropy')

        model = tflearn.DNN(net, tensorboard_verbose=0)

    model.fit(norm_train_images,
              npy_training_labels,
              n_epoch=number_of_epochs,
              shuffle=False,
              validation_set=(norm_test_images, npy_test_labels),
              show_metric=True,
              run_id='mlp')

    return model


def list_findings(labels, test_images, model):
    """Return lists of predicted false negative/positive labels/data."""
    npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
    npy_test_images = npy_test_images.astype(numpy.float32)
    npy_test_images = numpy.multiply(npy_test_images, 1.0 / 255.0)

    false_pos = []
    false_neg = []
    fp_images = []
    fn_images = []
    index = 0
    for x in range(0, len(npy_test_images) - 100, 100):
        images = npy_test_images[x:x + 100]
        image_tuples = test_images[x:x + 100]
        index, false_pos, false_neg, fp_images, fn_images = sort_findings(model,
                                                                          image_tuples,
                                                                          images,
                                                                          labels,
                                                                          false_pos,
                                                                          false_neg,
                                                                          fp_images,
                                                                          fn_images,
                                                                          index)
    images = npy_test_images[index:]
    image_tuples = test_images[index:]
    index, false_pos, false_neg, fp_images, fn_images = sort_findings(model,
                                                                      image_tuples,
                                                                      images,
                                                                      labels,
                                                                      false_pos,
                                                                      false_neg,
                                                                      fp_images,
                                                                      fn_images,
                                                                      index)

    return false_pos, false_neg, fp_images, fn_images


def sort_findings(model, image_tuples, test_images, labels, false_positives, false_negatives,
                  fp_images, fn_images, index):
    """False positive if model says road doesn't exist, but OpenStreetMap says it does.

    False negative if model says road exists, but OpenStreetMap doesn't list it.
    """
    pred_index = 0
    for p in model.predict(test_images):
        label = labels[index][0]
        if has_ways_in_center(label, 1) and p[0] > .5:
            false_positives.append(p)
            fp_images.append(image_tuples[pred_index])
        elif not has_ways_in_center(label, 16) and p[0] <= .5:
            false_negatives.append(p)
            fn_images.append(image_tuples[pred_index])
        pred_index += 1
        index += 1
    return index, false_positives, false_negatives, fp_images, fn_images


def predictions_for_tiles(test_images, model):
    """Batch predictions on the test image set, to avoid a memory spike."""
    npy_test_images = numpy.array([img_loc_tuple[0] for img_loc_tuple in test_images])
    test_images = npy_test_images.astype(numpy.float32)
    test_images = numpy.multiply(test_images, 1.0 / 255.0)

    all_predictions = []
    for x in range(0, len(test_images) - 100, 100):
        for p in model.predict(test_images[x:x + 100]):
            all_predictions.append(p)

    for p in model.predict(test_images[len(all_predictions):]):
        all_predictions.append(p)
    assert len(all_predictions) == len(test_images)

    return all_predictions
